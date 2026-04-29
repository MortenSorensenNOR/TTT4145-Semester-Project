#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// CRC lookup tables — built once at static-init time
// ---------------------------------------------------------------------------

static const std::array<uint8_t, 256> CRC8_TABLE = []() {
    std::array<uint8_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
        uint8_t crc = (uint8_t)i;
        for (int _ = 0; _ < 8; ++_)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
        t[i] = crc;
    }
    return t;
}();

static const std::array<uint16_t, 256> CRC16_TABLE = []() {
    std::array<uint16_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
        uint16_t crc = (uint16_t)(i << 8);
        for (int _ = 0; _ < 8; ++_)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
        t[i] = crc;
    }
    return t;
}();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline int write_bits(int n, int length, int* out, int offset) {
    for (int i = 0; i < length; ++i)
        out[offset + i] = (n >> (length - 1 - i)) & 1;
    return offset + length;
}

static inline int read_bits(const int* in, int& offset, int length) {
    int v = 0;
    for (int i = 0; i < length; ++i)
        v = (v << 1) | (in[offset + i] & 1);
    offset += length;
    return v;
}

static std::vector<int> int_to_bits(int n, int length) {
    std::vector<int> bits(length);
    write_bits(n, length, bits.data(), 0);
    return bits;
}

static int bits_to_int(const std::vector<int>& bits) {
    int result = 0;
    for (int b : bits)
        result = (result << 1) | (b & 1);
    return result;
}

// ---------------------------------------------------------------------------
// Enums / structs
// ---------------------------------------------------------------------------

enum class ModulationSchemes : int { BPSK = 0, QPSK = 1, PSK8 = 2, PSK16 = 3 };

struct FrameHeader {
    int               length = 0;
    int               src = 0;
    int               dst = 0;
    int               frame_type = 0;
    ModulationSchemes mod_scheme = ModulationSchemes::BPSK;
    int               sequence_number = 0;
    int               coding_rate = 4;  // CodeRates.FIVE_SIXTH_RATE.value
    int               crc = 0;
    bool              crc_passed = true;
};

struct FrameHeaderConfig {
    int  payload_length_bits  = 11;
    int  src_bits             = 2;
    int  dst_bits             = 2;
    int  frame_type_bits      = 2;
    int  mod_scheme_bits      = 3;
    int  sequence_number_bits = 7;
    int  coding_rate_bits     = 3;
    int  crc_bits             = 8;
    bool use_golay            = false;

    int header_total_size() const {
        return payload_length_bits + src_bits + dst_bits + frame_type_bits
             + mod_scheme_bits + sequence_number_bits + coding_rate_bits
             + crc_bits;
    }
};

// ---------------------------------------------------------------------------
// FrameHeaderConstructor
// ---------------------------------------------------------------------------

class FrameHeaderConstructor {
public:
    explicit FrameHeaderConstructor(const FrameHeaderConfig& cfg) : cfg_(cfg) {
        int raw = cfg.header_total_size();
        header_length_ = 2 * (int)std::ceil(raw / 2.0);
        data_field_bits_ = cfg.payload_length_bits + cfg.src_bits + cfg.dst_bits
                         + cfg.frame_type_bits + cfg.mod_scheme_bits
                         + cfg.sequence_number_bits + cfg.coding_rate_bits;
    }

    int header_length() const { return header_length_; }

    int encode_into(const FrameHeader& hdr, int* out) const {
        int off = 0;
        off = write_bits(hdr.length,                       cfg_.payload_length_bits,  out, off);
        off = write_bits(hdr.src,                          cfg_.src_bits,             out, off);
        off = write_bits(hdr.dst,                          cfg_.dst_bits,             out, off);
        off = write_bits(hdr.frame_type,                   cfg_.frame_type_bits,      out, off);
        off = write_bits(static_cast<int>(hdr.mod_scheme), cfg_.mod_scheme_bits,      out, off);
        off = write_bits(hdr.sequence_number,              cfg_.sequence_number_bits, out, off);
        off = write_bits(hdr.coding_rate,                  cfg_.coding_rate_bits,     out, off);
        uint8_t crc = crc8_from_bits(out, data_field_bits_);
        off = write_bits(crc, cfg_.crc_bits, out, off);
        return off;
    }

    std::vector<int> encode(const FrameHeader& hdr) const {
        std::vector<int> out(header_length_);
        encode_into(hdr, out.data());
        return out;
    }

    FrameHeader decode(const int* raw, int /*len*/) const {
        int off = 0;
        int length          = read_bits(raw, off, cfg_.payload_length_bits);
        int src             = read_bits(raw, off, cfg_.src_bits);
        int dst             = read_bits(raw, off, cfg_.dst_bits);
        int frame_type      = read_bits(raw, off, cfg_.frame_type_bits);
        int mod_val         = read_bits(raw, off, cfg_.mod_scheme_bits);
        int sequence_number = read_bits(raw, off, cfg_.sequence_number_bits);
        int coding_rate     = read_bits(raw, off, cfg_.coding_rate_bits);

        int data_end = off;
        int crc = read_bits(raw, off, cfg_.crc_bits);

        uint8_t expected_crc = crc8_from_bits(raw, data_end);

        FrameHeader h;
        h.length          = length;
        h.src             = src;
        h.dst             = dst;
        h.frame_type      = frame_type;
        h.mod_scheme      = static_cast<ModulationSchemes>(mod_val);
        h.sequence_number = sequence_number;
        h.coding_rate     = coding_rate;
        h.crc             = crc;
        h.crc_passed      = (crc == (int)expected_crc);
        return h;
    }

    FrameHeader decode(const std::vector<int>& raw) const {
        return decode(raw.data(), (int)raw.size());
    }

private:
    FrameHeaderConfig cfg_;
    int               header_length_;
    int               data_field_bits_;

    uint8_t crc8_from_bits(const int* bits, int n_bits) const {
        int total_bytes = (n_bits + 7) / 8;
        int pad         = total_bytes * 8 - n_bits;
        uint8_t crc = 0x00;
        int bi = 0;
        for (int byte_idx = 0; byte_idx < total_bytes; ++byte_idx) {
            uint8_t byte = 0;
            int start = (byte_idx == 0 && pad > 0) ? pad : 0;
            for (int b = start; b < 8; ++b)
                byte = (byte << 1) | (bits[bi++] & 1);
            crc = CRC8_TABLE[crc ^ byte];
        }
        return crc;
    }
};

// ---------------------------------------------------------------------------
// FrameConstructor
// ---------------------------------------------------------------------------

class FrameConstructor {
public:
    static constexpr int PAYLOAD_CRC_BITS     = 16;
    static constexpr int PAYLOAD_PAD_MULTIPLE = 12;

    explicit FrameConstructor(const FrameHeaderConfig& cfg = FrameHeaderConfig{})
        : hdr_cfg_(cfg), hdr_ctor_(cfg) {}

    int header_encoded_n_bits() const { return hdr_ctor_.header_length(); }

    int payload_coded_n_bits(const FrameHeader& hdr) const {
        int raw = hdr.length * 8 + PAYLOAD_CRC_BITS;
        return raw + ((-raw % PAYLOAD_PAD_MULTIPLE + PAYLOAD_PAD_MULTIPLE) % PAYLOAD_PAD_MULTIPLE);
    }

    std::pair<py::array_t<int>, py::array_t<int>>
    encode(const FrameHeader& hdr,
           py::array_t<int, py::array::c_style | py::array::forcecast> payload_np) const {
        auto       buf = payload_np.request();
        const int* src = static_cast<const int*>(buf.ptr);
        int        psz = (int)buf.size;

        // Header — write directly into output array
        py::array_t<int> header_arr(hdr_ctor_.header_length());
        hdr_ctor_.encode_into(hdr, static_cast<int*>(header_arr.request().ptr));

        // Payload + CRC + padding — single allocation, no intermediate copy
        int n = payload_coded_n_bits(hdr);
        py::array_t<int> payload_arr(n);
        int* dst = static_cast<int*>(payload_arr.request().ptr);

        std::copy(src, src + psz, dst);
        uint16_t crc = crc16_from_bits(src, psz);
        write_bits(crc, PAYLOAD_CRC_BITS, dst, psz);
        std::fill(dst + psz + PAYLOAD_CRC_BITS, dst + n, 0);

        return { std::move(header_arr), std::move(payload_arr) };
    }

    FrameHeader decode_header(
        py::array_t<int, py::array::c_style | py::array::forcecast> header_np) const {
        auto buf = header_np.request();
        FrameHeader hdr = hdr_ctor_.decode(static_cast<const int*>(buf.ptr), (int)buf.size);
        if (!hdr.crc_passed)
            throw std::runtime_error("Header did not yield valid crc");
        return hdr;
    }

    py::array_t<int> decode_payload(
        const FrameHeader& hdr,
        py::array_t<double, py::array::c_style | py::array::forcecast> payload_np,
        bool soft = false) const
    {
        auto          buf   = payload_np.request();
        const double* ptr   = static_cast<const double*>(buf.ptr);
        int           sz    = (int)buf.size;
        int           dlen  = hdr.length * 8;
        int           crc_end = dlen + PAYLOAD_CRC_BITS;

        if (sz < crc_end)
            throw std::runtime_error("Payload buffer too short");

        py::array_t<int> out(dlen);
        int* out_ptr = static_cast<int*>(out.request().ptr);

        // Decode bits and compute CRC-16 in a single pass — no intermediate vector
        uint32_t reg = 0xFFFFu;
        constexpr uint32_t mask = 0xFFFFu;
        constexpr uint32_t msb  = 0x8000u;

        for (int i = 0; i < dlen; ++i) {
            int bit = soft ? (ptr[i] < 0.0 ? 1 : 0) : (int)ptr[i];
            out_ptr[i] = bit;
            reg ^= (uint32_t)(bit & 1) << 15;
            reg = (reg & msb) ? ((reg << 1) ^ 0x1021u) & mask : (reg << 1) & mask;
        }
        uint16_t expected_crc = (uint16_t)reg;

        // Decode CRC field directly from the buffer
        uint16_t received_crc = 0;
        for (int i = dlen; i < crc_end; ++i) {
            int bit = soft ? (ptr[i] < 0.0 ? 1 : 0) : (int)ptr[i];
            received_crc = (uint16_t)((received_crc << 1) | (bit & 1));
        }

        if (received_crc != expected_crc) {
            std::ostringstream oss;
            oss << "Payload CRC-16 mismatch: got 0x" << std::hex << std::setw(4)
                << std::setfill('0') << received_crc
                << ", expected 0x" << std::setw(4) << expected_crc;
            throw std::runtime_error(oss.str());
        }
        return out;
    }

private:
    FrameHeaderConfig      hdr_cfg_;
    FrameHeaderConstructor hdr_ctor_;

    static uint16_t crc16_from_bits(const int* bits, int n_bits) {
        uint16_t crc = 0xFFFFu;
        int i = 0;
        int whole_bytes = n_bits / 8;
        for (int b = 0; b < whole_bytes; ++b) {
            uint8_t byte = 0;
            for (int k = 0; k < 8; ++k)
                byte = (byte << 1) | (bits[i++] & 1);
            crc = (uint16_t)((crc << 8) ^ CRC16_TABLE[((crc >> 8) ^ byte) & 0xFF]);
        }
        // Remaining bits
        int rem = n_bits % 8;
        for (int k = 0; k < rem; ++k) {
            int bit = bits[i++] & 1;
            crc ^= (uint16_t)(bit << 15);
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021u) : (uint16_t)(crc << 1);
        }
        return crc;
    }
};

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(frame_constructor_ext, m, py::mod_gil_not_used()) {
    m.doc() = "C++ frame constructor with pybind11 bindings (optimized)";

    py::enum_<ModulationSchemes>(m, "ModulationSchemes")
        .value("BPSK", ModulationSchemes::BPSK)
        .value("QPSK", ModulationSchemes::QPSK)
        .value("PSK8", ModulationSchemes::PSK8)
        .value("PSK16", ModulationSchemes::PSK16)
        .export_values();

    py::class_<FrameHeader>(m, "FrameHeader")
        .def(py::init<>())
        .def(py::init([](int length, int src, int dst, int frame_type,
                         ModulationSchemes mod, int seq, int coding_rate,
                         int crc, bool crc_ok) {
            FrameHeader h;
            h.length = length; h.src = src; h.dst = dst;
            h.frame_type = frame_type; h.mod_scheme = mod;
            h.sequence_number = seq; h.coding_rate = coding_rate;
            h.crc = crc; h.crc_passed = crc_ok;
            return h;
        }), py::arg("length"), py::arg("src"), py::arg("dst"),
            py::arg("frame_type"), py::arg("mod_scheme"),
            py::arg("sequence_number"),
            py::arg("coding_rate") = 4,
            py::arg("crc") = 0,
            py::arg("crc_passed") = true)
        .def_readwrite("length",          &FrameHeader::length)
        .def_readwrite("src",             &FrameHeader::src)
        .def_readwrite("dst",             &FrameHeader::dst)
        .def_readwrite("frame_type",      &FrameHeader::frame_type)
        .def_readwrite("mod_scheme",      &FrameHeader::mod_scheme)
        .def_readwrite("sequence_number", &FrameHeader::sequence_number)
        .def_readwrite("coding_rate",     &FrameHeader::coding_rate)
        .def_readwrite("crc",             &FrameHeader::crc)
        .def_readwrite("crc_passed",      &FrameHeader::crc_passed)
        .def("__repr__", [](const FrameHeader& h) {
            return "<FrameHeader length=" + std::to_string(h.length)
                 + " src=" + std::to_string(h.src)
                 + " dst=" + std::to_string(h.dst) + ">";
        });

    py::class_<FrameHeaderConfig>(m, "FrameHeaderConfig")
        .def(py::init<>())
        .def_readwrite("payload_length_bits",  &FrameHeaderConfig::payload_length_bits)
        .def_readwrite("src_bits",             &FrameHeaderConfig::src_bits)
        .def_readwrite("dst_bits",             &FrameHeaderConfig::dst_bits)
        .def_readwrite("frame_type_bits",      &FrameHeaderConfig::frame_type_bits)
        .def_readwrite("mod_scheme_bits",      &FrameHeaderConfig::mod_scheme_bits)
        .def_readwrite("sequence_number_bits", &FrameHeaderConfig::sequence_number_bits)
        .def_readwrite("coding_rate_bits",     &FrameHeaderConfig::coding_rate_bits)
        .def_readwrite("crc_bits",             &FrameHeaderConfig::crc_bits)
        .def_readwrite("use_golay",            &FrameHeaderConfig::use_golay)
        .def("header_total_size",              &FrameHeaderConfig::header_total_size);

    py::class_<FrameHeaderConstructor>(m, "FrameHeaderConstructor")
        .def(py::init<const FrameHeaderConfig&>(), py::arg("config"))
        .def("header_length", &FrameHeaderConstructor::header_length)
        .def("encode", [](const FrameHeaderConstructor& self, const FrameHeader& hdr) {
            auto bits = self.encode(hdr);
            py::array_t<int> arr(bits.size());
            std::copy(bits.begin(), bits.end(), static_cast<int*>(arr.request().ptr));
            return arr;
        }, py::arg("header"))
        .def("decode", [](const FrameHeaderConstructor& self,
                          py::array_t<int, py::array::c_style | py::array::forcecast> arr) {
            auto buf = arr.request();
            return self.decode(static_cast<const int*>(buf.ptr), (int)buf.size);
        }, py::arg("header"));

    py::class_<FrameConstructor>(m, "FrameConstructor")
        .def(py::init<const FrameHeaderConfig&>(),
             py::arg("header_config") = FrameHeaderConfig{})
        .def("header_encoded_n_bits", &FrameConstructor::header_encoded_n_bits)
        .def("payload_coded_n_bits",  &FrameConstructor::payload_coded_n_bits,
             py::arg("header"))
        .def("encode",         &FrameConstructor::encode,
             py::arg("header"), py::arg("payload"))
        .def("decode_header",  &FrameConstructor::decode_header,
             py::arg("header_encoded"))
        .def("decode_payload", &FrameConstructor::decode_payload,
             py::arg("header"), py::arg("payload_encoded"), py::arg("soft") = false);

    m.def("int_to_bits", &int_to_bits, py::arg("n"), py::arg("length"));
    m.def("bits_to_int", &bits_to_int, py::arg("bits"));
}
