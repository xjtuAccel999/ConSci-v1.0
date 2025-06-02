// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_MAT_H
#define NCNN_MAT_H

#include <stdlib.h>
#include <string.h>

#include "../config.h"

#include "allocator.h"
#include "option.h"
#include "platform.h"

namespace ncnn {

// the three dimension matrix
class   Mat
{
public:
    // empty
    Mat();
    // vec
    Mat(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    // image
    Mat(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    // dim
    Mat(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // cube
    Mat(int w, int h, int d, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // packed vec
    Mat(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed image
    Mat(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed dim
    Mat(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed cube
    Mat(int w, int h, int d, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // copy
    Mat(const Mat& m);
    // external vec
    Mat(int w, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external image
    Mat(int w, int h, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external dim
    Mat(int w, int h, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external cube
    Mat(int w, int h, int d, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external packed vec
    Mat(int w, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed image
    Mat(int w, int h, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed dim
    Mat(int w, int h, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed cube
    Mat(int w, int h, int d, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // release
    ~Mat();
    // assign
    Mat& operator=(const Mat& m);
    // set all
    void fill(float v);
    void fill(int v);

    template<typename T>
    void fill(T v);
    // deep copy
    Mat clone(Allocator* allocator = 0) const;
    // deep copy from other mat, inplace
    void clone_from(const ncnn::Mat& mat, Allocator* allocator = 0);
    // reshape vec
    Mat reshape(int w, Allocator* allocator = 0) const;
    // reshape image
    Mat reshape(int w, int h, Allocator* allocator = 0) const;
    // reshape dim
    Mat reshape(int w, int h, int c, Allocator* allocator = 0) const;
    // reshape cube
    Mat reshape(int w, int h, int d, int c, Allocator* allocator = 0) const;
    // allocate vec
    void create(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate image
    void create(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate cube
    void create(int w, int h, int d, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed cube
    void create(int w, int h, int d, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate like
    void create_like(const Mat& m, Allocator* allocator = 0);

    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // bits per element
    int elembits() const;

    // shape only
    Mat shape() const;

    // data reference
    Mat channel(int c);
    const Mat channel(int c) const;
    Mat depth(int z);
    const Mat depth(int z) const;
    float* row(int y);
    const float* row(int y) const;
    template<typename T>
    T* row(int y);
    template<typename T>
    const T* row(int y) const;

    // range reference
    Mat channel_range(int c, int channels);
    const Mat channel_range(int c, int channels) const;
    Mat depth_range(int z, int depths);
    const Mat depth_range(int z, int depths) const;
    Mat row_range(int y, int rows);
    const Mat row_range(int y, int rows) const;
    Mat range(int x, int n);
    const Mat range(int x, int n) const;

    // access raw data
    template<typename T>
    operator T*();
    template<typename T>
    operator const T*() const;

    // convenient access float vec element
    float& operator[](size_t i);
    const float& operator[](size_t i) const;

#if NCNN_PIXEL
    enum PixelType
    {
        PIXEL_CONVERT_SHIFT = 16,
        PIXEL_FORMAT_MASK = 0x0000ffff,
        PIXEL_CONVERT_MASK = 0xffff0000,

        PIXEL_RGB = 1,
        PIXEL_BGR = 2,
        PIXEL_GRAY = 3,
        PIXEL_RGBA = 4,
        PIXEL_BGRA = 5,

        PIXEL_RGB2BGR = PIXEL_RGB | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2GRAY = PIXEL_RGB | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2RGBA = PIXEL_RGB | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2BGRA = PIXEL_RGB | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_BGR2RGB = PIXEL_BGR | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2GRAY = PIXEL_BGR | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2RGBA = PIXEL_BGR | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2BGRA = PIXEL_BGR | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_GRAY2RGB = PIXEL_GRAY | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGR = PIXEL_GRAY | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2RGBA = PIXEL_GRAY | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGRA = PIXEL_GRAY | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_RGBA2RGB = PIXEL_RGBA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGR = PIXEL_RGBA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2GRAY = PIXEL_RGBA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGRA = PIXEL_RGBA | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_BGRA2RGB = PIXEL_BGRA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGRA2BGR = PIXEL_BGRA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_BGRA2GRAY = PIXEL_BGRA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_BGRA2RGBA = PIXEL_BGRA | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
    };
    // convenient construct from pixel data
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, Allocator* allocator = 0);
    // convenient construct from pixel data with stride(bytes-per-row) parameter
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, Allocator* allocator = 0);
    // convenient construct from pixel data and resize to specific size
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height, Allocator* allocator = 0);
    // convenient construct from pixel data and resize to specific size with stride(bytes-per-row) parameter
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator = 0);
    // convenient construct from pixel data roi
    static Mat from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, Allocator* allocator = 0);
    // convenient construct from pixel data roi with stride(bytes-per-row) parameter
    static Mat from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, Allocator* allocator = 0);
    // convenient construct from pixel data roi and resize to specific size
    static Mat from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator = 0);
    // convenient construct from pixel data roi and resize to specific size with stride(bytes-per-row) parameter
    static Mat from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator = 0);

    // convenient export to pixel data
    void to_pixels(unsigned char* pixels, int type) const;
    // convenient export to pixel data with stride(bytes-per-row) parameter
    void to_pixels(unsigned char* pixels, int type, int stride) const;
    // convenient export to pixel data and resize to specific size
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const;
    // convenient export to pixel data and resize to specific size with stride(bytes-per-row) parameter
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height, int target_stride) const;

#endif // NCNN_PIXEL

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

    // convenient construct from half precision floating point data
    static Mat from_float16(const unsigned short* data, int size);

    // pointer to the data
    void* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-d-h-w-1  c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-d-h-w-4  c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-d-h-w-8  c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    Allocator* allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int d;
    int c;

    size_t cstep;
};


// misc function
#if NCNN_PIXEL
// convert yuv420sp(nv21) to rgb, the fast approximate version
  void yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// convert yuv420sp(nv12) to rgb, the fast approximate version
  void yuv420sp2rgb_nv12(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// convert yuv420sp(nv21) to rgb with half resize, the faster approximate version
  void yuv420sp2rgb_half(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// image pixel bilinear resize
  void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
  void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
  void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
  void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
// image pixel bilinear resize with stride(bytes-per-row) parameter
  void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
  void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
  void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
  void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
// image pixel bilinear resize, convenient wrapper for yuv420sp(nv21/nv12)
  void resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
#endif // NCNN_PIXEL
#if NCNN_PIXEL_ROTATE
// type is the from type, 6 means rotating from 6 to 1
//
//     1        2       3      4         5            6           7          8
//
//   888888  888888      88  88      8888888888  88                  88  8888888888
//   88          88      88  88      88  88      88  88          88  88      88  88
//   8888      8888    8888  8888    88          8888888888  8888888888          88
//   88          88      88  88
//   88          88  888888  888888
//
// ref http://sylvana.net/jpegcrop/exif_orientation.html
// image pixel kanna rotate
  void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
  void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
  void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
  void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
// image pixel kanna rotate with stride(bytes-per-row) parameter
  void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
  void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
  void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
  void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
// image pixel kanna rotate, convenient wrapper for yuv420sp(nv21/nv12)
  void kanna_rotate_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
#endif // NCNN_PIXEL_ROTATE
#if NCNN_PIXEL_AFFINE
// resolve affine transform matrix from rotation angle, scale factor and x y offset
  void get_rotation_matrix(float angle, float scale, float dx, float dy, float* tm);
// resolve affine transform matrix from two set of points, num_point must be >= 2
  void get_affine_transform(const float* points_from, const float* points_to, int num_point, float* tm);
// resolve the inversion affine transform matrix
  void invert_affine_transform(const float* tm, float* tm_inv);
// image pixel bilinear warpaffine inverse transform, set -233 for transparent border color, the color RGBA is little-endian encoded
  void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
  void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
  void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
  void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
// image pixel bilinear warpaffine inverse transform with stride(bytes-per-row) parameter, set -233 for transparent border color, the color RGBA is little-endian encoded
  void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type = 0, unsigned int v = 0);
  void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type = 0, unsigned int v = 0);
  void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type = 0, unsigned int v = 0);
  void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type = 0, unsigned int v = 0);
// image pixel bilinear warpaffine, convenient wrapper for yuv420sp(nv21/nv12), set -233 for transparent border color, the color YUV_ is little-endian encoded
  void warpaffine_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type = 0, unsigned int v = 0);
#endif // NCNN_PIXEL_AFFINE
#if NCNN_PIXEL_DRAWING
// draw rectangle, set thickness -1 for filled rectangle, the color RGBA is little-endian encoded
  void draw_rectangle_c1(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
  void draw_rectangle_c2(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
  void draw_rectangle_c3(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
  void draw_rectangle_c4(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
// draw rectangle with stride(bytes-per-row) parameter, set thickness -1 for filled rectangle, the color RGBA is little-endian encoded
  void draw_rectangle_c1(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
  void draw_rectangle_c2(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
  void draw_rectangle_c3(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
  void draw_rectangle_c4(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
// draw rectangle, convenient wrapper for yuv420sp(nv21/nv12), set thickness -1 for filled rectangle, the color YUV_ is little-endian encoded
  void draw_rectangle_yuv420sp(unsigned char* yuv420sp, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
// draw circle, set thickness -1 for filled circle, the color RGBA is little-endian encoded
  void draw_circle_c1(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
  void draw_circle_c2(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
  void draw_circle_c3(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
  void draw_circle_c4(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
// draw circle with stride(bytes-per-row) parameter, set thickness -1 for filled circle, the color RGBA is little-endian encoded
  void draw_circle_c1(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
  void draw_circle_c2(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
  void draw_circle_c3(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
  void draw_circle_c4(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
// draw circle, convenient wrapper for yuv420sp(nv21/nv12), set thickness -1 for filled circle, the color YUV_ is little-endian encoded
  void draw_circle_yuv420sp(unsigned char* yuv420sp, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
// draw line, the color RGBA is little-endian encoded
  void draw_line_c1(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
  void draw_line_c2(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
  void draw_line_c3(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
  void draw_line_c4(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
// draw line with stride(bytes-per-row) parameter, the color RGBA is little-endian encoded
  void draw_line_c1(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
  void draw_line_c2(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
  void draw_line_c3(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
  void draw_line_c4(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
// draw line, convenient wrapper for yuv420sp(nv21/nv12), the color YUV_ is little-endian encoded
  void draw_line_yuv420sp(unsigned char* yuv420sp, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
// resolve text bounding box size
  void get_text_drawing_size(const char* text, int fontpixelsize, int* w, int* h);
// draw ascii printables and newline, the color RGBA is little-endian encoded
  void draw_text_c1(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
  void draw_text_c2(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
  void draw_text_c3(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
  void draw_text_c4(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
// draw ascii printables and newline with stride(bytes-per-row) parameter, the color RGBA is little-endian encoded
  void draw_text_c1(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
  void draw_text_c2(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
  void draw_text_c3(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
  void draw_text_c4(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
// draw ascii printables and newline, convenient wrapper for yuv420sp(nv21/nv12), the color YUV_ is little-endian encoded
  void draw_text_yuv420sp(unsigned char* yuv420sp, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
#endif // NCNN_PIXEL_DRAWING

// type conversion
// convert float to half precision floating point
  unsigned short float32_to_float16(float value);
// convert half precision floating point to float
  float float16_to_float32(unsigned short value);
// convert float to brain half
  NCNN_FORCEINLINE unsigned short float32_to_bfloat16(float value)
{
    // 16 : 16
    union
    {
        unsigned int u;
        float f;
    } tmp;
    tmp.f = value;
    return tmp.u >> 16;
}
// convert brain half to float
  NCNN_FORCEINLINE float bfloat16_to_float32(unsigned short value)
{
    // 16 : 16
    union
    {
        unsigned int u;
        float f;
    } tmp;
    tmp.u = value << 16;
    return tmp.f;
}

// mat process
enum BorderType
{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
    BORDER_REFLECT = 2,
    BORDER_TRANSPARENT = -233,
};
  void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt = Option());
  void copy_make_border_3d(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int front, int behind, int type, float v, const Option& opt = Option());
  void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const Option& opt = Option());
  void copy_cut_border_3d(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int front, int behind, const Option& opt = Option());
  void resize_nearest(const Mat& src, Mat& dst, int w, int h, const Option& opt = Option());
  void resize_bilinear(const Mat& src, Mat& dst, int w, int h, const Option& opt = Option());
  void resize_bicubic(const Mat& src, Mat& dst, int w, int h, const Option& opt = Option());
  void convert_packing(const Mat& src, Mat& dst, int elempack, const Option& opt = Option());
  void flatten(const Mat& src, Mat& dst, const Option& opt = Option());
  void cast_float32_to_float16(const Mat& src, Mat& dst, const Option& opt = Option());
  void cast_float16_to_float32(const Mat& src, Mat& dst, const Option& opt = Option());
  void cast_int8_to_float32(const Mat& src, Mat& dst, const Option& opt = Option());
  void cast_float32_to_bfloat16(const Mat& src, Mat& dst, const Option& opt = Option());
  void cast_bfloat16_to_float32(const Mat& src, Mat& dst, const Option& opt = Option());
  void quantize_to_int8(const Mat& src, Mat& dst, const Mat& scale_data, const Option& opt = Option());
  void dequantize_from_int32(const Mat& src, Mat& dst, const Mat& scale_data, const Mat& bias_data, const Option& opt = Option());
  void requantize_from_int32_to_int8(const Mat& src, Mat& dst, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt = Option());

NCNN_FORCEINLINE Mat::Mat()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
}

NCNN_FORCEINLINE Mat::Mat(int _w, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _d, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _d, _c, _elemsize, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), d(0), c(0), cstep(0)
{
    create(_w, _h, _d, _c, _elemsize, _elempack, _allocator);
}

NCNN_FORCEINLINE Mat::Mat(const Mat& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), d(m.d), c(m.c), cstep(m.cstep)
{
    addref();
}

NCNN_FORCEINLINE Mat::Mat(int _w, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = (size_t)w * h;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), d(1), c(_c)
{
    cstep = alignSize((size_t)w * h * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;
    // cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _d, int _c, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(4), w(_w), h(_h), d(_d), c(_c)
{
    cstep = alignSize((size_t)w * h * d * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;
    // cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE Mat::Mat(int _w, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), d(1), c(1)
{
    cstep = w;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), d(1), c(1)
{
    cstep = (size_t)w * h;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), d(1), c(_c)
{
    cstep = alignSize((size_t)w * h * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;
    // cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE Mat::Mat(int _w, int _h, int _d, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(4), w(_w), h(_h), d(_d), c(_c)
{
    cstep = alignSize((size_t)w * h * d * elemsize, NCNN_MAT_ALIGN_BYTES) / elemsize;
    // cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;
}

NCNN_FORCEINLINE Mat::~Mat()
{
    release();
}

NCNN_FORCEINLINE void Mat::fill(float _v)
{
    int size = (int)total();
    float* ptr = (float*)data;

    int i = 0;

    for (; i < size; i++)
    {
        *ptr++ = _v;
    }
}

NCNN_FORCEINLINE void Mat::fill(int _v)
{
    int size = (int)total();
    int* ptr = (int*)data;

    int i = 0;

    for (; i < size; i++)
    {
        *ptr++ = _v;
    }
}

template<typename T>
NCNN_FORCEINLINE void Mat::fill(T _v)
{
    int size = (int)total();
    T* ptr = (T*)data;
    for (int i = 0; i < size; i++)
    {
        ptr[i] = _v;
    }
}

NCNN_FORCEINLINE Mat& Mat::operator=(const Mat& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        NCNN_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    d = m.d;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

NCNN_FORCEINLINE void Mat::addref()
{
    if (refcount)
        NCNN_XADD(refcount, 1);
}

NCNN_FORCEINLINE void Mat::release()
{
    if (refcount && NCNN_XADD(refcount, -1) == 1)
    {
        if (allocator)
            allocator->fastFree(data);
        else
            fastFree(data);
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    d = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

NCNN_FORCEINLINE bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

NCNN_FORCEINLINE size_t Mat::total() const
{
    return cstep * c;
}

NCNN_FORCEINLINE int Mat::elembits() const
{
    return elempack ? static_cast<int>(elemsize * 8) / elempack : 0;
}

//return a mat with the same shape
NCNN_FORCEINLINE Mat Mat::shape() const
{
    if (dims == 1)
        return Mat(w * elempack, (void*)0);
    if (dims == 2)
        return Mat(w, h * elempack, (void*)0);
    if (dims == 3)
        return Mat(w, h, c * elempack, (void*)0);
    if (dims == 4)
        return Mat(w, h, d, c * elempack, (void*)0);

    return Mat();
}


NCNN_FORCEINLINE Mat Mat::channel(int _c)
{
    Mat m(w, h, d, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
    m.dims = dims - 1;
    if (dims == 4)
        m.cstep = (size_t)w * h;
    return m;
}

NCNN_FORCEINLINE const Mat Mat::channel(int _c) const
{
    Mat m(w, h, d, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
    m.dims = dims - 1;
    if (dims == 4)
        m.cstep = (size_t)w * h;
    return m;
}

NCNN_FORCEINLINE Mat Mat::depth(int z)
{
    return Mat(w, h, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE const Mat Mat::depth(int z) const
{
    return Mat(w, h, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE float* Mat::row(int y)
{
    return (float*)((unsigned char*)data + (size_t)w * y * elemsize);
}

NCNN_FORCEINLINE const float* Mat::row(int y) const
{
    return (const float*)((unsigned char*)data + (size_t)w * y * elemsize);
}

template<typename T>
NCNN_FORCEINLINE T* Mat::row(int y)
{
    return (T*)((unsigned char*)data + (size_t)w * y * elemsize);
}

template<typename T>
NCNN_FORCEINLINE const T* Mat::row(int y) const
{
    return (const T*)((unsigned char*)data + (size_t)w * y * elemsize);
}

NCNN_FORCEINLINE Mat Mat::channel_range(int _c, int channels)
{
    Mat m(w, h, d, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
    m.dims = dims;
    return m;
}

NCNN_FORCEINLINE const Mat Mat::channel_range(int _c, int channels) const
{
    Mat m(w, h, d, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
    m.dims = dims;
    return m;
}

NCNN_FORCEINLINE Mat Mat::depth_range(int z, int depths)
{
    Mat m(w, h, depths, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize, elempack, allocator);
    m.cstep = (size_t)w * h;
    return m;
}

NCNN_FORCEINLINE const Mat Mat::depth_range(int z, int depths) const
{
    Mat m(w, h, depths, (unsigned char*)data + (size_t)w * h * z * elemsize, elemsize, elempack, allocator);
    m.cstep = (size_t)w * h;
    return m;
}

NCNN_FORCEINLINE Mat Mat::row_range(int y, int rows)
{
    return Mat(w, rows, (unsigned char*)data + (size_t)w * y * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE const Mat Mat::row_range(int y, int rows) const
{
    return Mat(w, rows, (unsigned char*)data + (size_t)w * y * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE Mat Mat::range(int x, int n)
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

NCNN_FORCEINLINE const Mat Mat::range(int x, int n) const
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

template<typename T>
NCNN_FORCEINLINE Mat::operator T*()
{
    return (T*)data;
}

template<typename T>
NCNN_FORCEINLINE Mat::operator const T*() const
{
    return (const T*)data;
}

NCNN_FORCEINLINE float& Mat::operator[](size_t i)
{
    return ((float*)data)[i];
}

NCNN_FORCEINLINE const float& Mat::operator[](size_t i) const
{
    return ((const float*)data)[i];
}

} // namespace ncnn

#endif // NCNN_MAT_H
