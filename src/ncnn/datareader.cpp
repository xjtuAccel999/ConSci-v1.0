// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "datareader.h"
#include "allocator.h"
#include "assert.h"

#include "../config.h"

#include <string.h>

// unsigned char* test_ptr = NULL;
// int   test_first = 1;

namespace ncnn {

DataReader::DataReader()
{
}

DataReader::~DataReader()
{
}

#if NCNN_STRING
int DataReader::scan(const char* /*format*/, void* /*p*/) const
{
    return 0;
}
#endif // NCNN_STRING

size_t DataReader::read(void* /*buf*/, size_t /*size*/) const
{
    return 0;
}

size_t DataReader::reference(size_t /*size*/, const void** /*buf*/) const
{
    return 0;
}

#if NCNN_STDIO
class DataReaderFromStdioPrivate
{
public:
    DataReaderFromStdioPrivate(FILE* _fp)
        : fp(_fp)
    {
    }
    FILE* fp;
};

DataReaderFromStdio::DataReaderFromStdio(FILE* _fp)
    : DataReader(), d(new DataReaderFromStdioPrivate(_fp))
{
}

DataReaderFromStdio::~DataReaderFromStdio()
{
    delete d;
}

DataReaderFromStdio::DataReaderFromStdio(const DataReaderFromStdio&)
    : d(0)
{
}

DataReaderFromStdio& DataReaderFromStdio::operator=(const DataReaderFromStdio&)
{
    return *this;
}

#if NCNN_STRING
int DataReaderFromStdio::scan(const char* format, void* p) const
{
    return fscanf(d->fp, format, p);
}
#endif // NCNN_STRING

size_t DataReaderFromStdio::read(void* buf, size_t size) const
{
    return fread(buf, 1, size, d->fp);
}
#endif // NCNN_STDIO

class DataReaderFromMemoryPrivate
{
public:
    DataReaderFromMemoryPrivate(const unsigned char*& _mem)
        : mem(_mem)
    {
    }
    const unsigned char*& mem;
};

DataReaderFromMemory::DataReaderFromMemory(const unsigned char*& _mem)
    : DataReader(), d(new DataReaderFromMemoryPrivate(_mem))
{
}

DataReaderFromMemory::~DataReaderFromMemory()
{
    delete d;
}

DataReaderFromMemory::DataReaderFromMemory(const DataReaderFromMemory&)
    : d(0)
{
}

DataReaderFromMemory& DataReaderFromMemory::operator=(const DataReaderFromMemory&)
{
    return *this;
}

#if NCNN_STRING
int DataReaderFromMemory::scan(const char* format, void* p) const
{
    size_t fmtlen = strlen(format);

    char* format_with_n = new char[fmtlen + 4];
    sprintf(format_with_n, "%s%%n", format);

    int nconsumed = 0;
    int nscan = sscanf((const char*)d->mem, format_with_n, p, &nconsumed);
    d->mem += nconsumed;

    delete[] format_with_n;

    return nconsumed > 0 ? nscan : 0;
}
#endif // NCNN_STRING

size_t DataReaderFromMemory::read(void* buf, size_t size) const
{
    // if(test_first == 1){
    //     test_ptr = (unsigned char*)d->mem;
    //     test_first = 0;
    // }
    // printf("DataReaderFromMemory::read, ptr = %p, offset_ptr = %ld (%ld)\n",d->mem,d->mem-test_ptr,d->mem-test_ptr);
    memcpy(buf, d->mem, size);
    // d->mem += size;
    d->mem += alignSize(size,NCNN_MAT_ALIGN_BYTES);
    return size;
}

size_t DataReaderFromMemory::reference(size_t size, const void** buf) const
{
    // printf("DataReaderFromMemory::reference, ptr = %p, offset_ptr = %ld (%ld)\n",d->mem,d->mem-test_ptr,d->mem-test_ptr);
    *buf = d->mem;
    // d->mem += size;
    d->mem += alignSize(size,NCNN_MAT_ALIGN_BYTES);
    return size;
}

} // namespace ncnn
