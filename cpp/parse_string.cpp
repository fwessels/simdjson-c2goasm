#include <immintrin.h>

#define UNUSED

struct ParsedJson {
  void write_tape(uint64_t val, uint8_t c) {}
  uint8_t *string_buf; // should be at least bytecapacity
  uint8_t *current_string_buf_loc;
};

/* result might be undefined when input_num is zero */
static inline int trailingzeroes(uint64_t input_num) {
#ifdef __BMI__
	return _tzcnt_u64(input_num);
#else
#warning "BMI is missing?"
	return __builtin_ctzll(input_num);
#endif
}

#define memcpy(DEST,SRC,SIZE) for (int ii = 0; ii < SIZE; ii++) { ((uint8_t *)DEST)[ii] = ((uint8_t *)SRC)[ii]; }

const signed char digittoval[256] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,
    9,  -1, -1, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1};

// returns a value with the high 16 bits set if not valid
// otherwise returns the conversion of the 4 hex digits at src into the bottom 16 bits of the 32-bit
// return register
static inline uint32_t hex_to_u32_nocheck(const uint8_t *src) {// strictly speaking, static inline is a C-ism
  // all these will sign-extend the chars looked up, placing 1-bits into the high 28 bits of every
  // invalid value. After the shifts, this will *still* result in the outcome that the high 16 bits of any
  // value with any invalid char will be all 1's. We check for this in the caller.
  int32_t v1 = digittoval[src[0]];
  int32_t v2 = digittoval[src[1]];
  int32_t v3 = digittoval[src[2]];
  int32_t v4 = digittoval[src[3]];
  return static_cast<uint32_t>(v1 << 12 | v2 << 8 | v3 << 4 | v4);
}

// given a code point cp, writes to c
// the utf-8 code, outputting the length in
// bytes, if the length is zero, the code point
// is invalid
//
// This can possibly be made faster using pdep
// and clz and table lookups, but JSON documents
// have few escaped code points, and the following
// function looks cheap.
//
// Note: we assume that surrogates are treated separately
//
inline size_t codepoint_to_utf8(uint32_t cp, uint8_t *c) {
  if (cp <= 0x7F) {
    c[0] = cp;
    return 1; // ascii
  } if (cp <= 0x7FF) {
    c[0] = (cp >> 6) + 192;
    c[1] = (cp & 63) + 128;
    return 2; // universal plane
  //  Surrogates are treated elsewhere...
  //} //else if (0xd800 <= cp && cp <= 0xdfff) {
  //  return 0; // surrogates // could put assert here
  } else if (cp <= 0xFFFF) {
    c[0] = (cp >> 12) + 224;
    c[1] = ((cp >> 6) & 63) + 128;
    c[2] = (cp & 63) + 128;
    return 3;
  } else if (cp <= 0x10FFFF) { // if you know you have a valid code point, this is not needed
    c[0] = (cp >> 18) + 240;
    c[1] = ((cp >> 12) & 63) + 128;
    c[2] = ((cp >> 6) & 63) + 128;
    c[3] = (cp & 63) + 128;
    return 4;
  }
  // will return 0 when the code point was too large.
  return 0; // bad r
}

// begin copypasta
// These chars yield themselves: " \ /
// b -> backspace, f -> formfeed, n -> newline, r -> cr, t -> horizontal tab
// u not handled in this table as it's complex
static const uint8_t escape_map[256] = {
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0, // 0x0.
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0x22, 0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0x2f,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,

    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0, // 0x4.
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0x5c, 0, 0,    0, // 0x5.
    0, 0, 0x08, 0, 0,    0, 0x0c, 0, 0, 0, 0, 0, 0,    0, 0x0a, 0, // 0x6.
    0, 0, 0x0d, 0, 0x09, 0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0, // 0x7.

    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,

    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
};

// handle a unicode codepoint
// write appropriate values into dest
// src will advance 6 bytes or 12 bytes
// dest will advance a variable amount (return via pointer)
// return true if the unicode codepoint was valid
// We work in little-endian then swap at write time
static inline bool handle_unicode_codepoint(const uint8_t **src_ptr, uint8_t **dst_ptr, bool quote_within_twelve) {
  // hex_to_u32_nocheck fills high 16 bits of the return value with 1s if the
  // conversion isn't valid; we defer the check for this to inside the
  // multilingual plane check
  uint32_t code_point = hex_to_u32_nocheck(*src_ptr + 2);
  *src_ptr += 6;
  // check for low surrogate for characters outside the Basic
  // Multilingual Plane.
  if (code_point >= 0xd800 && code_point < 0xdc00) {
    if (quote_within_twelve || ((*src_ptr)[0] != '\\') || (*src_ptr)[1] != 'u') {
      return false;
    }
    uint32_t code_point_2 = hex_to_u32_nocheck(*src_ptr + 2);

    // if the first code point is invalid we will get here, as we will go past
    // the check for being outside the Basic Multilingual plane. If we don't
    // find a \u immediately afterwards we fail out anyhow, but if we do,
    // this check catches both the case of the first code point being invalid
    // or the second code point being invalid.
    if ((code_point | code_point_2) >> 16) {
        return false;
    }

    code_point =
        (((code_point - 0xd800) << 10) | (code_point_2 - 0xdc00)) + 0x10000;
    *src_ptr += 6;
  }
  size_t offset = codepoint_to_utf8(code_point, *dst_ptr);
  *dst_ptr += offset;
  return offset > 0;
}

bool parse_string(const uint8_t *src, uint8_t *dst, uint8_t **pcurrent_string_buf_loc) {
  const uint8_t *const start_of_string = dst;
  while (1) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
    // store to dest unconditionally - we can overwrite the bits we don't like
    // later
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), v);
    auto bs_bits =
        static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8('\\'))));
    auto quote_mask = _mm256_cmpeq_epi8(v, _mm256_set1_epi8('"'));
    auto quote_bits =
        static_cast<uint32_t>(_mm256_movemask_epi8(quote_mask));
    if(((bs_bits - 1) & quote_bits) != 0 ) {
      // we encountered quotes first. Move dst to point to quotes and exit

      // find out where the quote is...
      uint32_t quote_dist = trailingzeroes(quote_bits);

      // NULL termination is still handy if you expect all your strings to be NULL terminated?
      // It comes at a small cost
      dst[quote_dist] = 0;

      uint32_t str_length = (dst - start_of_string) + quote_dist;
      memcpy(*pcurrent_string_buf_loc,&str_length, sizeof(uint32_t));
      ///////////////////////
      // Above, check for overflow in case someone has a crazy string (>=4GB?)
      // But only add the overflow check when the document itself exceeds 4GB
      // Currently unneeded because we refuse to parse docs larger or equal to 4GB.
      ////////////////////////


      // we advance the point, accounting for the fact that we have a NULl termination
      *pcurrent_string_buf_loc = dst + quote_dist + 1;

#ifdef JSON_TEST_STRINGS // for unit testing
      foundString(buf + offset,start_of_string,pj.current_string_buf_loc - 1);
#endif // JSON_TEST_STRINGS
      return true;
    }
    if(((quote_bits - 1) & bs_bits ) != 0 ) {
      // find out where the backspace is
      uint32_t bs_dist = trailingzeroes(bs_bits);
      uint8_t escape_char = src[bs_dist + 1];
      // we encountered backslash first. Handle backslash
      if (escape_char == 'u') {
        // move src/dst up to the start; they will be further adjusted
        // within the unicode codepoint handling code.
        src += bs_dist;
        dst += bs_dist;
        uint32_t quote_dist = 32;
        if (quote_bits != 0) {
          quote_dist = trailingzeroes(quote_bits);
        } else if (bs_dist > 32 - 12) {
          // the 6 + 6 byte unicode sequences can escape to the next YMM word,
          // so reload with shift and recheck for possibly premature quote
          uint32_t shift = bs_dist - 20;
          __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src-bs_dist+shift));
          auto quote_mask = _mm256_cmpeq_epi8(v, _mm256_set1_epi8('"'));
          auto quote_bits = static_cast<uint32_t>(_mm256_movemask_epi8(quote_mask));
          if (quote_bits != 0) {
            quote_dist = trailingzeroes(quote_bits);
          }
          quote_dist += shift;
        }
        bool quote_within_six = (quote_dist - bs_dist < 6); // is there a quote within six bytes
        bool quote_within_twelve = (quote_dist - bs_dist < 12); // is there a quote within twelve bytes
        if (quote_within_six || !handle_unicode_codepoint(&src, &dst, quote_within_twelve)) {
#ifdef JSON_TEST_STRINGS // for unit testing
          foundBadString(buf + offset);
#endif // JSON_TEST_STRINGS
          return false;
        }
      } else {
        // simple 1:1 conversion. Will eat bs_dist+2 characters in input and
        // write bs_dist+1 characters to output
        // note this may reach beyond the part of the buffer we've actually
        // seen. I think this is ok
        uint8_t escape_result = escape_map[escape_char];
        if (escape_result == 0u) {
#ifdef JSON_TEST_STRINGS // for unit testing
          foundBadString(buf + offset);
#endif // JSON_TEST_STRINGS
          return false; // bogus escape value is an error
        }
        dst[bs_dist] = escape_result;
        src += bs_dist + 2;
        dst += bs_dist + 1;
      }
    } else {
      // they are the same. Since they can't co-occur, it means we encountered
      // neither.
      src += 32;
      dst += 32;
    }
  }
  // can't be reached
  return true;
}

/*
 * ORIGINAL
 *
 * bool parse_string(const uint8_t *buf, UNUSED size_t len,
 *                   ParsedJson &pj, UNUSED const uint32_t depth, uint32_t offset) {
 * #ifdef SIMDJSON_SKIPSTRINGPARSING // for performance analysis, it is sometimes useful to skip parsing
 *   pj.write_tape(0, '"');// don't bother with the string parsing at all
 *   return true; // always succeeds
 * #else
 *   pj.write_tape(pj.current_string_buf_loc - pj.string_buf, '"');
 *   const uint8_t *src = &buf[offset + 1]; // we know that buf at offset is a "
 *   uint8_t *dst = pj.current_string_buf_loc + sizeof(uint32_t);
 *   return parse_string_internal(src, dst, &pj.current_string_buf_loc);
 * #endif // SIMDJSON_SKIPSTRINGPARSING
 * }
 */
