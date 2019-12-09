#include <immintrin.h>

// a straightforward comparison of a mask against input. 5 uops; would be
// cheaper in AVX512.
inline uint64_t cmp_mask_against_input(__m256i input_lo,
                                              __m256i input_hi, __m256i mask) {
  __m256i cmp_res_0 = _mm256_cmpeq_epi8(input_lo, mask);
  uint64_t res_0 = static_cast<uint32_t>(_mm256_movemask_epi8(cmp_res_0));
  __m256i cmp_res_1 = _mm256_cmpeq_epi8(input_hi, mask);
  uint64_t res_1 = _mm256_movemask_epi8(cmp_res_1);
  return res_0 | (res_1 << 32);
}

// find all values less than or equal than the content of maxval (using unsigned arithmetic)
inline uint64_t unsigned_lteq_against_input(__m256i input_lo,
                                              __m256i input_hi, __m256i maxval) {
  __m256i cmp_res_0 = _mm256_cmpeq_epi8(_mm256_max_epu8(maxval,input_lo),maxval);
  uint64_t res_0 = static_cast<uint32_t>(_mm256_movemask_epi8(cmp_res_0));
  __m256i cmp_res_1 = _mm256_cmpeq_epi8(_mm256_max_epu8(maxval,input_hi),maxval);
  uint64_t res_1 = _mm256_movemask_epi8(cmp_res_1);
  return res_0 | (res_1 << 32);
}

// return both the quote mask (which is a half-open mask that covers the first quote
// in an unescaped quote pair and everything in the quote pair) and the quote bits, which are the simple
// unescaped quoted bits. We also update the prev_iter_inside_quote value to tell the next iteration
// whether we finished the final iteration inside a quote pair; if so, this inverts our behavior of
// whether we're inside quotes for the next iteration.
// Note that we don't do any error checking to see if we have backslash sequences outside quotes; these
// backslash sequences (of any length) will be detected elsewhere.
uint64_t find_quote_mask_and_bits(
    __m256i *pinput_lo, __m256i *pinput_hi, uint64_t odd_ends,
    uint64_t *prev_iter_inside_quote, uint64_t *quote_bits, uint64_t *error_mask) {
  __m256i input_lo = _mm256_loadu_si256(pinput_lo);
  __m256i input_hi = _mm256_loadu_si256(pinput_hi);
  *quote_bits =
      cmp_mask_against_input(input_lo, input_hi, _mm256_set1_epi8('"'));
  *quote_bits = *quote_bits & ~odd_ends;
  // remove from the valid quoted region the unescapted characters.
  uint64_t quote_mask = _mm_cvtsi128_si64(_mm_clmulepi64_si128(
      _mm_set_epi64x(0ULL, *quote_bits), _mm_set1_epi8(0xFF), 0));
  quote_mask ^= *prev_iter_inside_quote;
  // All Unicode characters may be placed within the
  // quotation marks, except for the characters that MUST be escaped:
  // quotation mark, reverse solidus, and the control characters (U+0000
  //through U+001F).
  // https://tools.ietf.org/html/rfc8259
  uint64_t unescaped = unsigned_lteq_against_input(input_lo, input_hi, _mm256_set1_epi8(0x1F));
  *error_mask |= quote_mask & unescaped;
  // right shift of a signed value expected to be well-defined and standard
  // compliant as of C++20,
  // John Regher from Utah U. says this is fine code
  *prev_iter_inside_quote =
      static_cast<uint64_t>(static_cast<int64_t>(quote_mask) >> 63);
  return quote_mask;
}
