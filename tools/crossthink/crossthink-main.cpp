#include "crossthink.h"
#include "crossthink-ui.h"

#include <cpp-httplib/httplib.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>

using json = crossthink_json;

struct options {
    crossthink_options session;
    std::string socket_a;
    std::string socket_b;
    std::string api_key;
    std::string mcp_config;
    std::string prompt;
    std::string host = "127.0.0.1";
    int port = 8090;
    int mcp_timeout = 60;
};

static uint64_t number(const char * text, uint64_t maximum) {
    char * end = nullptr;
    errno = 0;
    const auto value = std::strtoull(text, &end, 10);
    if (errno || end == text || *end || *text < '0' || *text > '9' || value > maximum) {
        throw std::invalid_argument(std::string("invalid number: ") + text);
    }
    return value;
}

static bool parse(int argc, char ** argv, options & opts) {
    enum {
        OPT_HOST = 256, OPT_PORT, OPT_CHUNK, OPT_SPLICE_MODE, OPT_SENTENCE_AFTER,
        OPT_ANSWER, OPT_SEED, OPT_TEMPERATURE, OPT_KEY, OPT_MCP_CONFIG, OPT_MCP_TIMEOUT,
        OPT_TOOL_TOKENS, OPT_TOOL_ROUNDS, OPT_LEGACY_SPLICE, OPT_LINK_QUANTUM, OPT_LINK_WAIT, OPT_PRIVATE_QUANTUM,
    };
    const struct option names[] = {
        {"socket-a", required_argument, nullptr, 'a'},
        {"socket-b", required_argument, nullptr, 'b'},
        {"prompt", required_argument, nullptr, 'p'},
        {"file", required_argument, nullptr, 'f'},
        {"host", required_argument, nullptr, OPT_HOST},
        {"port", required_argument, nullptr, OPT_PORT},
        {"legacy-splice", no_argument, nullptr, OPT_LEGACY_SPLICE},
        {"link-quantum", required_argument, nullptr, OPT_LINK_QUANTUM},
        {"private-quantum", required_argument, nullptr, OPT_PRIVATE_QUANTUM},
        {"link-wait-tokens", required_argument, nullptr, OPT_LINK_WAIT},
        {"splice-mode", required_argument, nullptr, OPT_SPLICE_MODE},
        {"max-segment-tokens", required_argument, nullptr, OPT_CHUNK},
        {"chunk-tokens", required_argument, nullptr, OPT_CHUNK},
        {"sentence-after", required_argument, nullptr, OPT_SENTENCE_AFTER},
        {"answer-tokens", required_argument, nullptr, OPT_ANSWER},
        {"seed", required_argument, nullptr, OPT_SEED},
        {"temperature", required_argument, nullptr, OPT_TEMPERATURE},
        {"api-key", required_argument, nullptr, OPT_KEY},
        {"mcp-config", required_argument, nullptr, OPT_MCP_CONFIG},
        {"mcp-timeout", required_argument, nullptr, OPT_MCP_TIMEOUT},
        {"tool-turn-tokens", required_argument, nullptr, OPT_TOOL_TOKENS},
        {"max-tool-rounds", required_argument, nullptr, OPT_TOOL_ROUNDS},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    bool has_prompt = false;
    int opt;
    while ((opt = getopt_long(argc, argv, "a:b:p:f:h", names, nullptr)) != -1) {
        switch (opt) {
            case 'a': opts.socket_a = optarg; break;
            case 'b': opts.socket_b = optarg; break;
            case 'p':
            case 'f': {
                if (has_prompt) {
                    throw std::invalid_argument("use --prompt or --file once");
                }
                has_prompt = true;
                if (opt == 'p') {
                    opts.prompt = optarg;
                } else if (std::string(optarg) == "-") {
                    opts.prompt.assign(std::istreambuf_iterator<char>(std::cin), {});
                    if (std::cin.bad()) {
                        throw std::runtime_error("cannot read standard input");
                    }
                } else {
                    std::ifstream input(optarg);
                    if (!input) {
                        throw std::runtime_error(std::string("cannot open prompt file: ") + optarg);
                    }
                    opts.prompt.assign(std::istreambuf_iterator<char>(input), {});
                    if (input.bad()) {
                        throw std::runtime_error("cannot read prompt file");
                    }
                }
                break;
            }
            case OPT_HOST: opts.host = optarg; break;
            case OPT_PORT: opts.port = static_cast<int>(number(optarg, 65535)); break;
            case OPT_LEGACY_SPLICE: opts.session.telepathy = false; break;
            case OPT_LINK_QUANTUM: opts.session.link_quantum = static_cast<int32_t>(number(optarg, 4096)); break;
            case OPT_PRIVATE_QUANTUM: opts.session.private_quantum = static_cast<int32_t>(number(optarg, 4096)); break;
            case OPT_LINK_WAIT: opts.session.link_wait_tokens = static_cast<int32_t>(number(optarg, 65536)); break;
            case OPT_CHUNK: opts.session.chunk_tokens = static_cast<int32_t>(number(optarg, 4096)); break;
            case OPT_SENTENCE_AFTER: opts.session.sentence_after = static_cast<int32_t>(number(optarg, 4096)); break;
            case OPT_SPLICE_MODE: {
                const std::string mode = optarg;
                if (mode != "paragraph" && mode != "fixed") {
                    throw std::invalid_argument("splice mode must be paragraph or fixed");
                }
                opts.session.paragraph_splice = mode == "paragraph";
                break;
            }
            case OPT_ANSWER: opts.session.answer_tokens = static_cast<int32_t>(number(optarg, 65536)); break;
            case OPT_SEED: opts.session.seed = static_cast<uint32_t>(number(optarg, UINT32_MAX - 1)); break;
            case OPT_KEY: opts.api_key = optarg; break;
            case OPT_MCP_CONFIG:
                if (!opts.mcp_config.empty() || !*optarg) {
                    throw std::invalid_argument("use --mcp-config once with a nonempty file path");
                }
                opts.mcp_config = optarg;
                break;
            case OPT_MCP_TIMEOUT: opts.mcp_timeout = static_cast<int>(number(optarg, 600)); break;
            case OPT_TOOL_TOKENS: opts.session.tool_tokens = static_cast<int32_t>(number(optarg, 16384)); break;
            case OPT_TOOL_ROUNDS: opts.session.max_tool_rounds = static_cast<int32_t>(number(optarg, 64)); break;
            case OPT_TEMPERATURE: {
                char * end = nullptr;
                errno = 0;
                opts.session.temperature = std::strtod(optarg, &end);
                if (errno || end == optarg || *end || !std::isfinite(opts.session.temperature) || opts.session.temperature < 0) {
                    throw std::invalid_argument("invalid temperature");
                }
                break;
            }
            case 'h':
                std::cout << "Usage: " << argv[0] << " -a SOCKET -b SOCKET [options]\n"
                    "Two agents with an optional shared reasoning channel and browser console.\n\n"
                    "  -a, --socket-a PATH       First llama-server socket\n"
                    "  -b, --socket-b PATH       Second llama-server socket\n"
                    "  -p, --prompt TEXT         Start thinking immediately (default: wait for browser)\n"
                    "  -f, --file PATH           Read initial prompt (- for stdin)\n"
                    "      --host HOST           Console address (default: 127.0.0.1)\n"
                    "      --port PORT           Console port (default: 8090)\n"
                    "      --link-quantum N      Shared reasoning quantum, 1..4096 tokens (default: 64)\n"
                    "      --private-quantum N   Private reasoning quantum, 1..4096 tokens (default: 256)\n"
                    "      --link-wait-tokens N   Maximum voluntary yield, 1..65536 tokens (default: 512)\n"
                    "      --legacy-splice       Restore covert reasoning exchange (default: explicit link)\n"
                    "      --splice-mode MODE    Legacy paragraph rendezvous or fixed intervals (default: paragraph)\n"
                    "      --max-segment-tokens N Legacy segment ceiling, 1..4096 (default: 512)\n"
                    "      --chunk-tokens N      Alias for --max-segment-tokens\n"
                    "      --sentence-after N    Legacy sentence threshold, 1..4096 tokens (default: 256)\n"
                    "      --answer-tokens N     Answer budget, 1..65536 (default: 1024)\n"
                    "      --seed N              Initial sampling seed (default: 42)\n"
                    "      --temperature N       Sampling temperature (default: 1.0)\n"
                    "      --api-key KEY         Bearer key for both model servers (default: none)\n"
                    "      --mcp-config FILE     JSON mcpServers with HTTP/HTTPS urls and optional headers\n"
                    "      --mcp-timeout N       MCP HTTP timeout, 1..600 seconds (default: 60)\n"
                    "      --tool-turn-tokens N  Private turn budget, 1..16384 (default: 2048)\n"
                    "                           Effective budget is max(N, --answer-tokens)\n"
                    "      --max-tool-rounds N   Consecutive tool round limit, 1..64 (default: 8)\n"
                    "  -h, --help                Show help\n";
                return false;
            default: throw std::invalid_argument("unknown option; use --help");
        }
    }
    if (!opts.session.chunk_tokens || !opts.session.sentence_after) {
        throw std::invalid_argument("segment ceiling and sentence threshold must be at least 1");
    }
    if (!opts.session.link_quantum || !opts.session.private_quantum || !opts.session.link_wait_tokens) {
        throw std::invalid_argument("reasoning quanta and yield token limit must be at least 1");
    }
    if (!opts.mcp_timeout || !opts.session.tool_tokens || !opts.session.max_tool_rounds) {
        throw std::invalid_argument("MCP timeout, tool turn budget and tool round limit must be at least 1");
    }
    if (optind != argc || opts.socket_a.empty() || opts.socket_b.empty() || opts.socket_a == opts.socket_b ||
            opts.host.empty() || !opts.port || (has_prompt && opts.prompt.empty())) {
        throw std::invalid_argument("two different sockets and a valid console address are required; use --help");
    }
    return true;
}

static std::vector<crossthink_mcp_config> load_mcp_config(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open MCP config: " + path);
    }
    std::string contents;
    char buffer[4096];
    while (input.read(buffer, sizeof(buffer)) || input.gcount()) {
        const size_t size = static_cast<size_t>(input.gcount());
        if (contents.size() + size > 1024 * 1024) {
            throw std::invalid_argument("MCP config exceeds 1 MiB");
        }
        contents.append(buffer, size);
    }
    if (input.bad()) {
        throw std::runtime_error("cannot read MCP config: " + path);
    }
    return parse_crossthink_mcp_config(json::parse(contents));
}

static void routes(httplib::Server & http, crossthink_session & session) {
    http.set_payload_max_length(1024 * 1024 + 4096);
    http.Get("/", [](const httplib::Request &, httplib::Response & response) {
        response.set_header("Cache-Control", "no-store");
        response.set_content(crossthink_ui, "text/html; charset=utf-8");
    });
    http.Get("/state", [&](const httplib::Request &, httplib::Response & response) {
        response.set_header("Cache-Control", "no-store");
        response.set_content(session.state().dump(), "application/json");
    });
    for (const std::string action : {"message", "pause", "resume", "answer", "reset", "link_on", "link_off"}) {
        http.Post("/" + action, [&, action](const httplib::Request & request, httplib::Response & response) {
            try {
                const auto type = request.get_header_value("Content-Type");
                if (type.substr(0, type.find(';')) != "application/json") {
                    throw std::invalid_argument("Content-Type must be application/json");
                }
                const auto body = json::parse(request.body);
                if (!body.is_object()) {
                    throw std::invalid_argument("expected a JSON object");
                }
                session.command(action, body.value("text", std::string()));
                response.status = 202;
                response.set_content("{\"ok\":true}", "application/json");
            } catch (const std::exception & error) {
                response.status = 400;
                response.set_content(json{{"error", error.what()}}.dump(), "application/json");
            }
        });
    }
    http.Get("/events", [&](const httplib::Request & request, httplib::Response & response) {
        uint64_t cursor = 0;
        try {
            if (request.has_param("after")) {
                cursor = number(request.get_param_value("after").c_str(), UINT64_MAX - 1);
            }
        } catch (const std::exception & error) {
            response.status = 400;
            response.set_content(json{{"error", error.what()}}.dump(), "application/json");
            return;
        }
        if (cursor > session.state().at("last_event_id").get<uint64_t>()) {
            cursor = 0;
        }
        response.set_header("Cache-Control", "no-cache");
        response.set_header("X-Accel-Buffering", "no");
        response.set_chunked_content_provider("text/event-stream", [&, cursor](size_t, httplib::DataSink & sink) mutable {
            if (!sink.is_writable()) {
                return false;
            }
            const auto events = session.events_after(cursor, true);
            std::string data;
            for (const auto & event : events) {
                cursor = event.at("id").get<uint64_t>();
                data += "id: " + std::to_string(cursor) + "\ndata: " + event.dump() + "\n\n";
            }
            if (data.empty()) {
                data = ": keepalive\n\n";
            }
            return sink.write(data.data(), data.size());
        });
    });
}

int main(int argc, char ** argv) {
    try {
        options opts;
        if (!parse(argc, argv, opts)) {
            return 0;
        }
        sigset_t signals;
        sigemptyset(&signals);
        sigaddset(&signals, SIGINT);
        sigaddset(&signals, SIGTERM);
        if (pthread_sigmask(SIG_BLOCK, &signals, nullptr)) {
            throw std::runtime_error("cannot initialize signal handling");
        }
        std::array<std::unique_ptr<crossthink_transport>, 2> transports = {
            crossthink_unix_transport(opts.socket_a, opts.api_key),
            crossthink_unix_transport(opts.socket_b, opts.api_key),
        };
        std::array<std::unique_ptr<crossthink_tool_service>, 2> tool_services;
        if (!opts.mcp_config.empty()) {
            const auto configs = load_mcp_config(opts.mcp_config);
            for (auto & service : tool_services) {
                service = crossthink_mcp_service(configs, opts.mcp_timeout);
                if (!opts.session.telepathy && !opts.session.paragraph_splice && !service->tools().empty()) {
                    throw std::invalid_argument("MCP tools require --splice-mode paragraph");
                }
            }
        }
        crossthink_session session(std::move(transports), opts.session, std::move(tool_services));
        httplib::Server http;
        routes(http, session);
        if (!http.bind_to_port(opts.host, opts.port)) {
            throw std::runtime_error("cannot bind console address");
        }
        if (!opts.prompt.empty()) {
            session.command("message", opts.prompt);
        }
        std::cout << "Crossthink console: http://" << opts.host << ':' << opts.port << "/\n" << std::flush;
        std::thread shutdown([&] {
            int received;
            sigwait(&signals, &received);
            http.stop();
        });
        const bool ok = http.listen_after_bind();
        pthread_kill(shutdown.native_handle(), SIGTERM);
        shutdown.join();
        return ok ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "llama-crossthink: " << error.what() << '\n';
        return 1;
    }
}
