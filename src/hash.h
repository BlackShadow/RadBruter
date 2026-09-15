#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#else
#include <openssl/evp.h>
#endif

namespace lm {
class Hash {
#ifdef _WIN32
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE context = nullptr;
#else
    EVP_MD_CTX *context = nullptr;
#endif

  public:
    Hash() {
#ifdef _WIN32
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
            throw std::runtime_error("SHA-256 initialization failed");
        if (BCryptCreateHash(algorithm, &context, nullptr, 0, nullptr, 0, 0) < 0) {
            BCryptCloseAlgorithmProvider(algorithm, 0);
            throw std::runtime_error("SHA-256 initialization failed");
        }
#else
        context = EVP_MD_CTX_new();
        if (!context || EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
            EVP_MD_CTX_free(context);
            throw std::runtime_error("SHA-256 initialization failed");
        }
#endif
    }
    ~Hash() {
#ifdef _WIN32
        BCryptDestroyHash(context);
        BCryptCloseAlgorithmProvider(algorithm, 0);
#else
        EVP_MD_CTX_free(context);
#endif
    }
    Hash(const Hash &) = delete;
    Hash &operator=(const Hash &) = delete;
    void add(const void *data, size_t size) {
#ifdef _WIN32
        auto bytes = static_cast<const unsigned char *>(data);
        while (size) {
            auto count = static_cast<ULONG>(std::min(size, size_t(UINT32_MAX)));
            if (BCryptHashData(context, const_cast<PUCHAR>(bytes), count, 0) < 0)
                throw std::runtime_error("SHA-256 update failed");
            bytes += count;
            size -= count;
        }
#else
        if (EVP_DigestUpdate(context, data, size) != 1)
            throw std::runtime_error("SHA-256 update failed");
#endif
    }
    template <class T> void add(const T &value) { add(&value, sizeof value); }
    std::string finish() {
        unsigned char bytes[32];
#ifdef _WIN32
        if (BCryptFinishHash(context, bytes, sizeof bytes, 0) < 0)
            throw std::runtime_error("SHA-256 finish failed");
#else
        unsigned count;
        if (EVP_DigestFinal_ex(context, bytes, &count) != 1 || count != sizeof bytes)
            throw std::runtime_error("SHA-256 finish failed");
#endif
        const char *hex = "0123456789abcdef";
        std::string result;
        for (auto byte : bytes) {
            result += hex[byte >> 4];
            result += hex[byte & 15];
        }
        return result;
    }
};
} // namespace lm
