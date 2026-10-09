#include "server-token-wire.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <getopt.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

static constexpr char reasoning_open[] = "<think>";
static constexpr char reasoning_close[] = "</think>";

struct options {
    std::string socket_a;
    std::string socket_b;
    std::string prompt;
    int32_t reasoning_tokens = 4096;
    int32_t answer_tokens = 1024;
    uint32_t seed = 42;
    double temperature = 1.0;
};

static void usage(const char * program) {
    std::cout << "Usage: " << program << " -a SOCKET -b SOCKET {-p PROMPT | -f FILE} [options]\n"
              << "Generate two complete reasoning blocks, swap their raw token IDs, then answer.\n\n"
              << "  -a, --socket-a PATH         First llama-server Unix socket\n"
              << "  -b, --socket-b PATH         Second llama-server Unix socket\n"
              << "  -p, --prompt TEXT           Shared user prompt\n"
              << "  -f, --file PATH             Read prompt from a file (- for standard input)\n"
              << "  -n, --reasoning-tokens N    Reasoning token limit per peer (default: 4096)\n"
              << "      --answer-tokens N       Answer token limit per peer (default: 1024)\n"
              << "      --seed N                First sampling seed (default: 42; other stages use N+1..3)\n"
              << "      --temperature N         Sampling temperature (default: 1.0)\n"
              << "  -h, --help                  Show this help\n\n"
              << "Both servers must use the same tokenizer and a template ending in <think>.\n"
              << "Incomplete reasoning is rejected. Each answer sees only the other peer's reasoning.\n";
}

static uint32_t parse_integer(const char * value, const char * name, uint32_t maximum, bool allow_zero = false) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long number = std::strtoull(value, &end, 10);
    if (errno || value == end || *end || *value < '0' || *value > '9' || number > maximum || (!number && !allow_zero)) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<uint32_t>(number);
}

static std::string read_prompt(const char * path) {
    if (std::string(path) == "-") {
        std::string result(std::istreambuf_iterator<char>(std::cin), {});
        if (std::cin.bad()) {
            throw std::runtime_error("failed to read prompt from standard input");
        }
        return result;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("cannot open prompt file: ") + path);
    }
    std::string result(std::istreambuf_iterator<char>(input), {});
    if (input.bad()) {
        throw std::runtime_error(std::string("failed to read prompt file: ") + path);
    }
    return result;
}

static bool parse_options(int argc, char ** argv, struct options & opts) {
    enum { OPT_ANSWER_TOKENS = 256, OPT_SEED, OPT_TEMPERATURE };
    static const struct option long_options[] = {
        { "socket-a",         required_argument, nullptr, 'a' },
        { "socket-b",         required_argument, nullptr, 'b' },
        { "prompt",           required_argument, nullptr, 'p' },
        { "file",             required_argument, nullptr, 'f' },
        { "reasoning-tokens", required_argument, nullptr, 'n' },
        { "answer-tokens",    required_argument, nullptr, OPT_ANSWER_TOKENS },
        { "seed",             required_argument, nullptr, OPT_SEED },
        { "temperature",      required_argument, nullptr, OPT_TEMPERATURE },
        { "help",             no_argument,       nullptr, 'h' },
        { nullptr,            0,                 nullptr, 0 },
    };

    bool has_prompt = false;
    int opt;
    while ((opt = getopt_long(argc, argv, "a:b:p:f:n:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'a': opts.socket_a = optarg; break;
            case 'b': opts.socket_b = optarg; break;
            case 'p':
            case 'f':
                if (has_prompt) {
                    throw std::invalid_argument("specify exactly one --prompt or --file");
                }
                has_prompt = true;
                opts.prompt = opt == 'p' ? optarg : read_prompt(optarg);
                break;
            case 'n':
                opts.reasoning_tokens = parse_integer(optarg, "reasoning token limit", INT32_MAX);
                break;
            case OPT_ANSWER_TOKENS:
                opts.answer_tokens = parse_integer(optarg, "answer token limit", INT32_MAX);
                break;
            case OPT_SEED:
                opts.seed = parse_integer(optarg, "seed", UINT32_MAX - 4, true);
                break;
            case OPT_TEMPERATURE: {
                char * end = nullptr;
                errno = 0;
                opts.temperature = std::strtod(optarg, &end);
                if (errno || optarg == end || *end || !std::isfinite(opts.temperature) || opts.temperature < 0) {
                    throw std::invalid_argument(std::string("invalid temperature: ") + optarg);
                }
                break;
            }
            case 'h': usage(argv[0]); return false;
            default: throw std::invalid_argument("invalid option; use --help");
        }
    }
    if (optind != argc || opts.socket_a.empty() || opts.socket_b.empty() || !has_prompt || opts.prompt.empty()) {
        throw std::invalid_argument("two sockets and a nonempty prompt are required; use --help");
    }
    if (opts.socket_a == opts.socket_b) {
        throw std::invalid_argument("use two different server sockets");
    }
    return true;
}

struct completion {
    json metadata;
    std::vector<llama_token> tokens;
    double wall_ms;
};

struct peer {
    std::string label;
    httplib::Client client;
    int32_t n_vocab = 0;
    uint64_t context_size = 0;
    std::vector<llama_token> eog_ids;
    std::vector<llama_token> base;
    llama_token close_token = -1;

    peer(const char * label, const std::string & socket) : label(label), client(socket, 80) {
        client.set_address_family(AF_UNIX);
        client.set_connection_timeout(5);
        client.set_read_timeout(3600);
        client.set_write_timeout(30);
        client.set_keep_alive(true);
    }

    std::string body(const httplib::Result & result, const char * endpoint) const {
        if (!result) {
            throw std::runtime_error(label + " " + endpoint + ": " + httplib::to_string(result.error()));
        }
        if (result->status != 200) {
            throw std::runtime_error(label + " " + endpoint + ": HTTP " + std::to_string(result->status) + " " + result->body);
        }
        return result->body;
    }

    json post_json(const char * endpoint, const json & data) {
        return json::parse(body(client.Post(endpoint, data.dump(), "application/json"), endpoint));
    }

    json info() {
        json result = json::parse(body(client.Get("/tokens/info"), "/tokens/info"));
        const int64_t size = result.at("n_vocab").get<int64_t>();
        const int64_t context = result.at("context_size").get<int64_t>();
        if (result.at("protocol") != server_token_wire::magic || size <= 0 || size > INT32_MAX || context <= 0) {
            throw std::runtime_error(label + ": invalid token protocol or vocabulary/context size");
        }
        n_vocab = static_cast<int32_t>(size);
        context_size = static_cast<uint64_t>(context);
        eog_ids = read_tokens(result.at("eog_ids"));
        return result;
    }

    std::vector<llama_token> read_tokens(const json & data) const {
        if (!data.is_array()) {
            throw std::runtime_error(label + ": expected token array in bootstrap response");
        }
        std::vector<llama_token> result;
        result.reserve(data.size());
        for (const auto & value : data) {
            if (!value.is_number_integer()) {
                throw std::runtime_error(label + ": noninteger token in bootstrap response");
            }
            const int64_t id = value.get<int64_t>();
            if (id < 0 || id >= n_vocab) {
                throw std::runtime_error(label + ": token outside vocabulary in bootstrap response");
            }
            result.push_back(static_cast<llama_token>(id));
        }
        return result;
    }

    std::vector<llama_token> tokenize(const std::string & content, bool add_special) {
        const json result = post_json("/tokenize", {{"content", content}, {"add_special", add_special}, {"parse_special", true}});
        return read_tokens(result.at("tokens"));
    }

    void prepare(const struct options & opts) {
        const json result = post_json("/apply-template", {
            {"messages", json::array({{{"role", "user"}, {"content", opts.prompt}}})},
            {"chat_template_kwargs", {{"enable_thinking", true}}},
        });
        const std::string prompt = result.at("prompt").get<std::string>();
        const size_t opening = prompt.rfind(reasoning_open);
        if (opening == std::string::npos || prompt.find_first_not_of(" \t\r\n", opening + sizeof(reasoning_open) - 1) != std::string::npos) {
            throw std::runtime_error(label + ": chat template must end with an open <think> section; use --jinja and a reasoning-capable template");
        }
        base = tokenize(prompt, true);
        const auto closing = tokenize(reasoning_close, false);
        if (base.empty() || closing.size() != 1) {
            throw std::runtime_error(label + ": empty base prompt or </think> does not encode as one token");
        }
        close_token = closing.front();
        if (std::find(eog_ids.begin(), eog_ids.end(), close_token) != eog_ids.end()) {
            throw std::runtime_error(label + ": </think> cannot also be an end-of-generation token");
        }
        const uint64_t maximum = static_cast<uint64_t>(base.size()) + opts.reasoning_tokens + opts.answer_tokens;
        if (maximum >= context_size) {
            throw std::runtime_error(label + ": prompt plus reasoning/answer limits must fit below the server's per-slot context size");
        }
    }

    struct completion complete(const std::vector<llama_token> & prompt, const json & parameters) {
        const auto start = std::chrono::steady_clock::now();
        const std::string request = server_token_wire::encode(parameters.dump(), prompt);
        auto result = server_token_wire::decode(body(client.Post("/completion/tokens", request, "application/octet-stream"), "/completion/tokens"), n_vocab);
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return {json::parse(result.metadata), std::move(result.tokens), elapsed};
    }

    void validate_reasoning(const struct completion & result) const {
        if (result.metadata.at("stop_type") != "word" || result.metadata.at("stopping_word") != reasoning_close ||
                result.metadata.at("truncated").get<bool>() || result.tokens.empty() || result.tokens.back() != close_token) {
            throw std::runtime_error(label + ": reasoning did not finish at </think>; increase --reasoning-tokens if it hit the limit");
        }
        if (std::find(result.tokens.begin(), result.tokens.end() - 1, close_token) != result.tokens.end() - 1) {
            throw std::runtime_error(label + ": reasoning contains an earlier </think> token");
        }
        for (llama_token token : result.tokens) {
            if (std::find(eog_ids.begin(), eog_ids.end(), token) != eog_ids.end()) {
                throw std::runtime_error(label + ": reasoning contains an end-of-generation token");
            }
        }
    }
};

static std::array<struct completion, 2> complete_pair(struct peer & a, struct peer & b,
        const std::vector<llama_token> & prompt_a, const std::vector<llama_token> & prompt_b,
        const json & parameters_a, const json & parameters_b) {
    auto complete = [&](struct peer & target, const std::vector<llama_token> & prompt, const json & parameters) {
        try {
            return target.complete(prompt, parameters);
        } catch (...) {
            a.client.stop();
            b.client.stop();
            throw;
        }
    };
    auto result_a = std::async(std::launch::async, [&] { return complete(a, prompt_a, parameters_a); });
    auto result_b = std::async(std::launch::async, [&] { return complete(b, prompt_b, parameters_b); });
    return {result_a.get(), result_b.get()};
}

static void print_timing(const char * label, const char * stage, const struct completion & result) {
    std::cerr << label << ' ' << stage << ": " << result.tokens.size() << " tokens, " << result.wall_ms << " ms wall";
    if (result.metadata.contains("timings")) {
        std::cerr << ", " << result.metadata.at("timings").dump();
    }
    std::cerr << '\n';
}

static int run(const struct options & opts) {
    struct peer a("A", opts.socket_a);
    struct peer b("B", opts.socket_b);
    const json info_a = a.info();
    const json info_b = b.info();
    if (a.n_vocab != b.n_vocab || info_a.at("fingerprint") != info_b.at("fingerprint") || a.eog_ids != b.eog_ids) {
        throw std::runtime_error("peers have incompatible tokenizer mappings or end-of-generation tokens");
    }
    a.prepare(opts);
    b.prepare(opts);
    if (a.close_token != b.close_token) {
        throw std::runtime_error("peers disagree on the </think> token ID");
    }

    json parameters_a = {
        {"n_predict", opts.reasoning_tokens}, {"seed", opts.seed}, {"temperature", opts.temperature},
        {"reasoning_budget_tokens", -1},
        {"cache_prompt", true}, {"return_content", false}, {"stop", json::array({reasoning_close})},
        {"preserved_tokens", json::array({reasoning_close})},
    };
    json parameters_b = parameters_a;
    parameters_b["seed"] = opts.seed + 1;
    std::cerr << "Generating complete reasoning blocks on A and B...\n";
    const auto reasoning = complete_pair(a, b, a.base, b.base, parameters_a, parameters_b);
    a.validate_reasoning(reasoning[0]);
    b.validate_reasoning(reasoning[1]);
    print_timing("A", "reasoning", reasoning[0]);
    print_timing("B", "reasoning", reasoning[1]);

    // Replace each peer's reasoning; retain the closing marker to begin its answer.
    std::vector<llama_token> prompt_a = a.base;
    std::vector<llama_token> prompt_b = b.base;
    prompt_a.insert(prompt_a.end(), reasoning[1].tokens.begin(), reasoning[1].tokens.end());
    prompt_b.insert(prompt_b.end(), reasoning[0].tokens.begin(), reasoning[0].tokens.end());
    parameters_a = {
        {"n_predict", opts.answer_tokens}, {"seed", opts.seed + 2}, {"temperature", opts.temperature},
        {"reasoning_budget_tokens", -1},
        {"cache_prompt", true}, {"stop", json::array()},
    };
    parameters_b = parameters_a;
    parameters_b["seed"] = opts.seed + 3;
    std::cerr << "Swapping reasoning token blocks; generating both answers...\n";
    const auto answers = complete_pair(a, b, prompt_a, prompt_b, parameters_a, parameters_b);
    const char * labels[] = {"A", "B"};
    int status = 0;
    for (size_t i = 0; i < answers.size(); ++i) {
        const auto & answer = answers[i];
        print_timing(labels[i], "answer", answer);
        std::cout << "=== " << labels[i] << " (with " << labels[1 - i] << "'s reasoning) ===\n";
        if (answer.metadata.value("content_omitted", false)) {
            std::cout << "[Answer text exceeded the response metadata limit.]\n";
            std::cerr << labels[i] << ": cannot display answer; reduce --answer-tokens\n";
            status = 1;
        } else {
            std::cout << answer.metadata.at("content").get<std::string>() << '\n';
        }
        if (answer.metadata.at("stop_type") == "limit" || answer.metadata.at("truncated").get<bool>()) {
            std::cerr << labels[i] << ": answer reached its token/context limit\n";
        }
    }
    return status;
}

int main(int argc, char ** argv) {
    try {
        struct options opts;
        if (!parse_options(argc, argv, opts)) {
            return 0;
        }
        return run(opts);
    } catch (const std::exception & error) {
        std::cerr << "llama-reasoning-swap: " << error.what() << '\n';
        return 1;
    }
}
