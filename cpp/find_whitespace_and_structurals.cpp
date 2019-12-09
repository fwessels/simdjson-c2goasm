#include <immintrin.h>

void find_whitespace_and_structurals(__m256i *pinput_lo,
                                           __m256i *pinput_hi,
                                           uint64_t *whitespace,
                                           uint64_t *structurals) {
  __m256i input_lo = _mm256_loadu_si256(pinput_lo);
  __m256i input_hi = _mm256_loadu_si256(pinput_hi);
  // do a 'shufti' to detect structural JSON characters
  // they are { 0x7b } 0x7d : 0x3a [ 0x5b ] 0x5d , 0x2c
  // these go into the first 3 buckets of the comparison (1/2/4)

  // we are also interested in the four whitespace characters
  // space 0x20, linefeed 0x0a, horizontal tab 0x09 and carriage return 0x0d
  // these go into the next 2 buckets of the comparison (8/16)
  const __m256i low_nibble_mask = _mm256_setr_epi8(
      16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0,
      0, 8, 12, 1, 2, 9, 0, 0);
  const __m256i high_nibble_mask = _mm256_setr_epi8(
      8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 8, 0, 18, 4, 0, 1, 0, 1,
      0, 0, 0, 3, 2, 1, 0, 0);

  __m256i structural_shufti_mask = _mm256_set1_epi8(0x7);
  __m256i whitespace_shufti_mask = _mm256_set1_epi8(0x18);

  __m256i v_lo = _mm256_and_si256(
      _mm256_shuffle_epi8(low_nibble_mask, input_lo),
      _mm256_shuffle_epi8(high_nibble_mask,
                          _mm256_and_si256(_mm256_srli_epi32(input_lo, 4),
                                           _mm256_set1_epi8(0x7f))));

  __m256i v_hi = _mm256_and_si256(
      _mm256_shuffle_epi8(low_nibble_mask, input_hi),
      _mm256_shuffle_epi8(high_nibble_mask,
                          _mm256_and_si256(_mm256_srli_epi32(input_hi, 4),
                                           _mm256_set1_epi8(0x7f))));
  __m256i tmp_lo = _mm256_cmpeq_epi8(
      _mm256_and_si256(v_lo, structural_shufti_mask), _mm256_set1_epi8(0));
  __m256i tmp_hi = _mm256_cmpeq_epi8(
      _mm256_and_si256(v_hi, structural_shufti_mask), _mm256_set1_epi8(0));

  uint64_t structural_res_0 =
      static_cast<uint32_t>(_mm256_movemask_epi8(tmp_lo));
  uint64_t structural_res_1 = _mm256_movemask_epi8(tmp_hi);
  *structurals = ~(structural_res_0 | (structural_res_1 << 32));

  __m256i tmp_ws_lo = _mm256_cmpeq_epi8(
      _mm256_and_si256(v_lo, whitespace_shufti_mask), _mm256_set1_epi8(0));
  __m256i tmp_ws_hi = _mm256_cmpeq_epi8(
      _mm256_and_si256(v_hi, whitespace_shufti_mask), _mm256_set1_epi8(0));

  uint64_t ws_res_0 = static_cast<uint32_t>(_mm256_movemask_epi8(tmp_ws_lo));
  uint64_t ws_res_1 = _mm256_movemask_epi8(tmp_ws_hi);
  *whitespace = ~(ws_res_0 | (ws_res_1 << 32));
}
