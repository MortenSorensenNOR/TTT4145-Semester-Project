#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<int> int_to_bits(int n, int length) {
    std::vector<int> bits(length);
    for (int i = 0; i < length; ++i)
        bits[i] = (n >> (length - 1 - i)) & 1;
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

enum class ModulationSchemes : int {
    BPSK  = 0,
    QPSK  = 1,
    PSK8  = 2,
};

struct FrameHeader {
    int               length;           // payload length in bytes
    int               src;
    int               dst;
    int               frame_type;
    ModulationSchemes mod_scheme;
    int               sequence_number;
    int               crc         = 0;
    bool              crc_passed  = true;
};

struct FrameHeaderConfig {
    int  payload_length_bits   = 12;
    int  src_bits              = 2;
    int  dst_bits              = 2;
    int  frame_type_bits       = 2;
    int  mod_scheme_bits       = 2;
    int  sequence_number_bits  = 4;
    int  reserved_bits         = 4;
    int  crc_bits              = 8;
    bool use_golay             = false;

    int header_total_size() const {
        return payload_length_bits + src_bits + dst_bits + frame_type_bits
             + mod_scheme_bits + sequence_number_bits + reserved_bits + crc_bits;
    }
};

// ---------------------------------------------------------------------------
// FrameHeaderConstructor
// ---------------------------------------------------------------------------

class FrameHeaderConstructor {
public:
    explicit FrameHeaderConstructor(const FrameHeaderConfig& cfg)
        : cfg_(cfg)
    {
        int raw = cfg.header_total_size();
        // round up to next even number
        header_length_ = 2 * (int)std::ceil(raw / 2.0);
    }

    int header_length() const { return header_length_; }

    // Returns a flat vector<int> of bits
    std::vector<int> encode(const FrameHeader& hdr) const {
        auto bits = build_data_bits(hdr);
        uint8_t crc = crc_calc(bits);
        auto crc_bits = int_to_bits(crc, cfg_.crc_bits);
        bits.insert(bits.end(), crc_bits.begin(), crc_bits.end());
        return bits;
    }

    FrameHeader decode(const std::vector<int>& raw) const {
        int offset = 0;

        auto take = [&](int n) {
            std::vector<int> v(raw.begin() + offset, raw.begin() + offset + n);
            offset += n;
            return v;
        };

        auto length_bits          = take(cfg_.payload_length_bits);
        auto src_bits             = take(cfg_.src_bits);
        auto dst_bits             = take(cfg_.dst_bits);
        auto frame_type_bits      = take(cfg_.frame_type_bits);
        auto mod_scheme_bits      = take(cfg_.mod_scheme_bits);
        auto sequence_number_bits = take(cfg_.sequence_number_bits);

        offset += cfg_.reserved_bits;   // skip reserved
        auto crc_field = take(cfg_.crc_bits);

        int length          = bits_to_int(length_bits);
        int src             = bits_to_int(src_bits);
        int dst             = bits_to_int(dst_bits);
        int frame_type      = bits_to_int(frame_type_bits);
        int mod_val         = bits_to_int(mod_scheme_bits);
        int sequence_number = bits_to_int(sequence_number_bits);
        int crc             = bits_to_int(crc_field);

        // Recompute CRC over the data portion (no reserved, no crc field)
        auto data_bits = build_data_bits_from_fields(
            length_bits, src_bits, dst_bits,
            frame_type_bits, mod_scheme_bits, sequence_number_bits);
        uint8_t expected_crc = crc_calc(data_bits);
        auto expected_bits   = int_to_bits(expected_crc, cfg_.crc_bits);

        FrameHeader h;
        h.length          = length;
        h.src             = src;
        h.dst             = dst;
        h.frame_type      = frame_type;
        h.mod_scheme      = static_cast<ModulationSchemes>(mod_val);
        h.sequence_number = sequence_number;
        h.crc             = crc;
        h.crc_passed      = (crc_field == expected_bits);
        return h;
    }

private:
    FrameHeaderConfig cfg_;
    int               header_length_;

    // Builds the data bits (everything except CRC)
    std::vector<int> build_data_bits(const FrameHeader& hdr) const {
        auto lb  = int_to_bits(hdr.length,                       cfg_.payload_length_bits);
        auto sb  = int_to_bits(hdr.src,                          cfg_.src_bits);
        auto db  = int_to_bits(hdr.dst,                          cfg_.dst_bits);
        auto ftb = int_to_bits(hdr.frame_type,                   cfg_.frame_type_bits);
        auto msb = int_to_bits(static_cast<int>(hdr.mod_scheme), cfg_.mod_scheme_bits);
        auto snb = int_to_bits(hdr.sequence_number,              cfg_.sequence_number_bits);
        return build_data_bits_from_fields(lb, sb, db, ftb, msb, snb);
    }

    std::vector<int> build_data_bits_from_fields(
        const std::vector<int>& lb,
        const std::vector<int>& sb,
        const std::vector<int>& db,
        const std::vector<int>& ftb,
        const std::vector<int>& msb,
        const std::vector<int>& snb) const
    {
        std::vector<int> out;
        auto app = [&](const std::vector<int>& v){ out.insert(out.end(), v.begin(), v.end()); };
        app(lb); app(sb); app(db); app(ftb); app(msb); app(snb);
        out.insert(out.end(), cfg_.reserved_bits, 0);
        return out;
    }

    // CRC-8 (poly 0x07)
    uint8_t crc_calc(const std::vector<int>& data_bits) const {
        // Pad to multiple of 8
        int total = (int)data_bits.size();
        int padded_len = ((total + 7) / 8) * 8;
        std::vector<int> padded(padded_len - total, 0);
        padded.insert(padded.end(), data_bits.begin(), data_bits.end());

        uint8_t crc = 0x00;
        for (int i = 0; i < padded_len; i += 8) {
            uint8_t byte = 0;
            for (int b = 0; b < 8; ++b)
                byte = (byte << 1) | padded[i + b];
            crc ^= byte;
            for (int _ = 0; _ < 8; ++_) {
                if (crc & 0x80)
                    crc = (uint8_t)((crc << 1) ^ 0x07);
                else
                    crc = (uint8_t)(crc << 1);
            }
        }
        return crc;
    }
};

// ---------------------------------------------------------------------------
// FrameConstructor
// ---------------------------------------------------------------------------

class FrameConstructor {
public:
    static constexpr int PAYLOAD_CRC_BITS    = 16;
    static constexpr int PAYLOAD_PAD_MULTIPLE = 12;

    explicit FrameConstructor(const FrameHeaderConfig& cfg = FrameHeaderConfig{})
        : hdr_cfg_(cfg), hdr_ctor_(cfg) {}

    int header_encoded_n_bits() const {
        // Without Golay, 1:1 ratio
        return hdr_ctor_.header_length();
    }

    int payload_coded_n_bits(const FrameHeader& hdr) const {
        int raw = hdr.length * 8 + PAYLOAD_CRC_BITS;
        return raw + ((-raw) % PAYLOAD_PAD_MULTIPLE + PAYLOAD_PAD_MULTIPLE) % PAYLOAD_PAD_MULTIPLE;
    }

    // Returns (header_encoded, payload_encoded) as numpy int arrays
    std::pair<py::array_t<int>, py::array_t<int>>
    encode(const FrameHeader& hdr, py::array_t<int> payload_np) const {
        auto buf = payload_np.request();
        std::vector<int> payload(static_cast<int*>(buf.ptr),
                                 static_cast<int*>(buf.ptr) + buf.size);

        // Header
        std::vector<int> header_bits = hdr_ctor_.encode(hdr);

        // CRC-16 over payload
        uint16_t crc = crc16(payload);
        auto crc_bits = int_to_bits(crc, PAYLOAD_CRC_BITS);

        std::vector<int> payload_with_crc = payload;
        payload_with_crc.insert(payload_with_crc.end(), crc_bits.begin(), crc_bits.end());

        int n = payload_coded_n_bits(hdr);
        payload_with_crc.resize(n, 0);

        return { vec_to_array(header_bits), vec_to_array(payload_with_crc) };
    }

    FrameHeader decode_header(py::array_t<int> header_encoded_np) const {
        auto buf = header_encoded_np.request();
        std::vector<int> bits(static_cast<int*>(buf.ptr),
                              static_cast<int*>(buf.ptr) + buf.size);

        FrameHeader hdr = hdr_ctor_.decode(bits);
        if (!hdr.crc_passed)
            throw std::runtime_error("Header did not yield valid crc");
        return hdr;
    }

    py::array_t<int> decode_payload(const FrameHeader& hdr,
                                    py::array_t<double> payload_encoded_np,
                                    bool soft = false) const {
        auto buf = payload_encoded_np.request();
        auto* ptr = static_cast<double*>(buf.ptr);
        int sz = (int)buf.size;

        std::vector<int> payload_bits(sz);
        if (soft) {
            for (int i = 0; i < sz; ++i)
                payload_bits[i] = (ptr[i] < 0.0) ? 1 : 0;
        } else {
            for (int i = 0; i < sz; ++i)
                payload_bits[i] = (int)ptr[i];
        }

        int data_len   = hdr.length * 8;
        int crc_end    = data_len + PAYLOAD_CRC_BITS;

        std::vector<int> data_bits(payload_bits.begin(), payload_bits.begin() + data_len);
        std::vector<int> crc_bits (payload_bits.begin() + data_len, payload_bits.begin() + crc_end);

        uint16_t received_crc = (uint16_t)bits_to_int(crc_bits);
        uint16_t expected_crc = crc16(data_bits);

        if (received_crc != expected_crc) {
            std::ostringstream oss;
            oss << "Payload CRC-16 mismatch: got 0x" << std::hex << std::setw(4)
                << std::setfill('0') << received_crc
                << ", expected 0x" << std::setw(4) << expected_crc;
            throw std::runtime_error(oss.str());
        }
        return vec_to_array(data_bits);
    }

private:
    FrameHeaderConfig    hdr_cfg_;
    FrameHeaderConstructor hdr_ctor_;

    // CRC-16-CCITT (poly 0x1021, init 0xFFFF)
    static uint16_t crc16(const std::vector<int>& bits) {
        constexpr int    n    = PAYLOAD_CRC_BITS;
        constexpr uint32_t mask = (1u << n) - 1;
        constexpr uint32_t msb  = 1u << (n - 1);
        uint32_t reg = mask;
        for (int bit : bits) {
            reg ^= (uint32_t)(bit & 1) << (n - 1);
            if (reg & msb)
                reg = ((reg << 1) ^ 0x1021u) & mask;
            else
                reg = (reg << 1) & mask;
        }
        return (uint16_t)reg;
    }

    static py::array_t<int> vec_to_array(const std::vector<int>& v) {
        py::array_t<int> arr(v.size());
        auto buf = arr.request();
        std::copy(v.begin(), v.end(), static_cast<int*>(buf.ptr));
        return arr;
    }
};

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(frame_constructor, m) {
    m.doc() = "C++ frame constructor with pybind11 bindings";

    // --- ModulationSchemes ---
    py::enum_<ModulationSchemes>(m, "ModulationSchemes")
        .value("BPSK", ModulationSchemes::BPSK)
        .value("QPSK", ModulationSchemes::QPSK)
        .value("PSK8", ModulationSchemes::PSK8)
        .export_values();

    // --- FrameHeader ---
    py::class_<FrameHeader>(m, "FrameHeader")
        .def(py::init<>())
        .def(py::init([](int length, int src, int dst, int frame_type,
                         ModulationSchemes mod, int seq, int crc, bool crc_ok) {
            FrameHeader h;
            h.length = length; h.src = src; h.dst = dst;
            h.frame_type = frame_type; h.mod_scheme = mod;
            h.sequence_number = seq; h.crc = crc; h.crc_passed = crc_ok;
            return h;
        }), py::arg("length"), py::arg("src"), py::arg("dst"),
            py::arg("frame_type"), py::arg("mod_scheme"),
            py::arg("sequence_number"), py::arg("crc") = 0,
            py::arg("crc_passed") = true)
        .def_readwrite("length",          &FrameHeader::length)
        .def_readwrite("src",             &FrameHeader::src)
        .def_readwrite("dst",             &FrameHeader::dst)
        .def_readwrite("frame_type",      &FrameHeader::frame_type)
        .def_readwrite("mod_scheme",      &FrameHeader::mod_scheme)
        .def_readwrite("sequence_number", &FrameHeader::sequence_number)
        .def_readwrite("crc",             &FrameHeader::crc)
        .def_readwrite("crc_passed",      &FrameHeader::crc_passed)
        .def("__repr__", [](const FrameHeader& h) {
            return "<FrameHeader length=" + std::to_string(h.length)
                 + " src=" + std::to_string(h.src)
                 + " dst=" + std::to_string(h.dst) + ">";
        });

    // --- FrameHeaderConfig ---
    py::class_<FrameHeaderConfig>(m, "FrameHeaderConfig")
        .def(py::init<>())
        .def_readwrite("payload_length_bits",  &FrameHeaderConfig::payload_length_bits)
        .def_readwrite("src_bits",             &FrameHeaderConfig::src_bits)
        .def_readwrite("dst_bits",             &FrameHeaderConfig::dst_bits)
        .def_readwrite("frame_type_bits",      &FrameHeaderConfig::frame_type_bits)
        .def_readwrite("mod_scheme_bits",      &FrameHeaderConfig::mod_scheme_bits)
        .def_readwrite("sequence_number_bits", &FrameHeaderConfig::sequence_number_bits)
        .def_readwrite("reserved_bits",        &FrameHeaderConfig::reserved_bits)
        .def_readwrite("crc_bits",             &FrameHeaderConfig::crc_bits)
        .def_readwrite("use_golay",            &FrameHeaderConfig::use_golay)
        .def("header_total_size",              &FrameHeaderConfig::header_total_size);

    // --- FrameHeaderConstructor ---
    py::class_<FrameHeaderConstructor>(m, "FrameHeaderConstructor")
        .def(py::init<const FrameHeaderConfig&>(), py::arg("config"))
        .def("header_length", &FrameHeaderConstructor::header_length)
        .def("encode", [](const FrameHeaderConstructor& self, const FrameHeader& hdr) {
            auto bits = self.encode(hdr);
            py::array_t<int> arr(bits.size());
            std::copy(bits.begin(), bits.end(), static_cast<int*>(arr.request().ptr));
            return arr;
        }, py::arg("header"))
        .def("decode", [](const FrameHeaderConstructor& self, py::array_t<int> arr) {
            auto buf = arr.request();
            std::vector<int> bits(static_cast<int*>(buf.ptr),
                                  static_cast<int*>(buf.ptr) + buf.size);
            return self.decode(bits);
        }, py::arg("header"));

    // --- FrameConstructor ---
    py::class_<FrameConstructor>(m, "FrameConstructor")
        .def(py::init<const FrameHeaderConfig&>(),
             py::arg("header_config") = FrameHeaderConfig{})
        .def("header_encoded_n_bits", &FrameConstructor::header_encoded_n_bits)
        .def("payload_coded_n_bits",  &FrameConstructor::payload_coded_n_bits,
             py::arg("header"))
        .def("encode", &FrameConstructor::encode,
             py::arg("header"), py::arg("payload"))
        .def("decode_header", &FrameConstructor::decode_header,
             py::arg("header_encoded"))
        .def("decode_payload", &FrameConstructor::decode_payload,
             py::arg("header"), py::arg("payload_encoded"), py::arg("soft") = false);

    // --- Free helpers (mirror the Python module-level functions) ---
    m.def("int_to_bits", &int_to_bits, py::arg("n"), py::arg("length"));
    m.def("bits_to_int", &bits_to_int, py::arg("bits"));
}
