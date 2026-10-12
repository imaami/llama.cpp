#ifndef PTQ1_0_GLSL
#define PTQ1_0_GLSL

// The 28-byte block has 24 qs bytes, two qh bytes and one f16 scale.
// Four-byte loads preserve its stride and share packed bytes between trits.
struct block_ptq1_0_packed32 {
    uint qs[6];
    uint qh_d;
};
layout (binding = 0) readonly buffer A_PTQ1_0_PACKED32 {
    block_ptq1_0_packed32 data_a_ptq1_0_packed32[];
};

const uint ptq1_0_pow3[5] = uint[5](1u, 3u, 9u, 27u, 81u);

// Skip n recurrence steps in two independent 16-bit lanes. The largest
// product is 255*81 = 20655, so neither lane carries into the next one.
uint ptq1_0_lane_trits(uint bytes, uint n) {
    const uint v = (bytes * ptq1_0_pow3[n]) & 0x00FF00FFu;
    return ((v * 3u) >> 8u) & 0x00030003u;
}

float ptq1_0_trit(uint ib, uint a_offset, uint e) {
    uint b;
    uint n;
    if (e < 80u) {
        b = uint(data_a[a_offset + ib].qs[e & 15u]);
        n = e >> 4u;
    } else if (e < 120u) {
        const uint t = e - 80u;
        b = uint(data_a[a_offset + ib].qs[16u + (t & 7u)]);
        n = t >> 3u;
    } else {
        const uint t = e - 120u;
        b = uint(data_a[a_offset + ib].qh[t & 1u]);
        n = t >> 1u;
    }

    const uint v = (b * ptq1_0_pow3[n]) & 0xFFu;
    return float(int((v * 3u) >> 8u) - 1);
}

// Main bytes interleave 16, then eight elements per trit level. The final
// qh pair interleaves two elements per level into positions 120..127.
uint ptq1_0_word(uint ib, uint a_offset, uint e, out uint n) {
    if (e < 80u) {
        n = e >> 4u;
        return data_a_ptq1_0_packed32[a_offset + ib].qs[(e & 15u) >> 2u];
    }
    if (e < 120u) {
        const uint t = e - 80u;
        n = t >> 3u;
        return data_a_ptq1_0_packed32[a_offset + ib].qs[4u + ((t & 7u) >> 2u)];
    }
    n = (e - 120u) >> 1u;
    return data_a_ptq1_0_packed32[a_offset + ib].qh_d;
}

vec2 ptq1_0_trits2(uint ib, uint a_offset, uint e) {
    if ((e & 1u) != 0u) {
        return vec2(ptq1_0_trit(ib, a_offset, e), ptq1_0_trit(ib, a_offset, e + 1u));
    }
    uint n;
    const uint w = ptq1_0_word(ib, a_offset, e, n) >> (e < 120u ? 8u * (e & 2u) : 0u);
    const uint q = ptq1_0_lane_trits((w & 0xFFu) | ((w & 0xFF00u) << 8u), n);
    return vec2(q & 3u, q >> 16u) - 1.0f;
}

vec4 ptq1_0_trits4(uint ib, uint a_offset, uint e) {
    if ((e & 3u) != 0u) {
        return vec4(ptq1_0_trit(ib, a_offset, e), ptq1_0_trit(ib, a_offset, e + 1u),
                    ptq1_0_trit(ib, a_offset, e + 2u), ptq1_0_trit(ib, a_offset, e + 3u));
    }
    uint n;
    const uint w = ptq1_0_word(ib, a_offset, e, n);
    if (e < 120u) {
        const uint qe = ptq1_0_lane_trits(w & 0x00FF00FFu, n);
        const uint qo = ptq1_0_lane_trits((w >> 8u) & 0x00FF00FFu, n);
        return vec4(qe & 3u, qo & 3u, qe >> 16u, qo >> 16u) - 1.0f;
    }
    const uint bytes = (w & 0xFFu) | ((w & 0xFF00u) << 8u);
    const uint q0 = ptq1_0_lane_trits(bytes, n);
    const uint q1 = ptq1_0_lane_trits(bytes, n + 1u);
    return vec4(q0 & 3u, q0 >> 16u, q1 & 3u, q1 >> 16u) - 1.0f;
}

#endif // PTQ1_0_GLSL
