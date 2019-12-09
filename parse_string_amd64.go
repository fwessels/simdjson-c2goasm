//+build !noasm
//+build !appengine

package simdjson

import (
	"unsafe"
	"reflect"
)

//go:noescape
func _parse_string(src, dst, pcurrent_string_buf_loc unsafe.Pointer)

////go:noescape
//func _parse_string2(src, string_buf, dummy_loc unsafe.Pointer) uint64

func parse_string_simd(buf []byte, stringbuf *[]byte) int {

	sh := (*reflect.SliceHeader)(unsafe.Pointer(stringbuf))

	string_buf_loc := uintptr(unsafe.Pointer(sh.Data)) + uintptr(sh.Len)
	src := uintptr(unsafe.Pointer(&buf[0])) + 1 // const uint8_t *src = &buf[offset + 1];
	dst := string_buf_loc + 4                   // uint8_t *dst = pj.current_string_buf_loc + sizeof(uint32_t);

	_parse_string(unsafe.Pointer(src), unsafe.Pointer(dst), unsafe.Pointer(&string_buf_loc))

	written := int(uintptr(string_buf_loc) - (dst - 4))
	if sh.Len + written >= sh.Cap {
		panic("Memory corruption -- written beyond slice capacity -- expected capacity to be larger than max values written")
	}
	sh.Len += written

	return written
}
//
//func parse_string_simd2(buf []byte, stringbuf *[]byte) int {
//
//	src := uintptr(unsafe.Pointer(&buf[0])) + 1 // const uint8_t *src = &buf[offset + 1];
//	dummy := uint64(0)
//
//	written := _parse_string2(unsafe.Pointer(src), unsafe.Pointer(stringbuf), unsafe.Pointer(&dummy))
//
//	return int(written)
//}
