from bit import bit_reverse
import benchmark
from testing import assert_equal
from random import rand

fn CRC32(data: List[UInt8], dummy_table: List[UInt32]) -> UInt32:
    var crc32: UInt32 = 0xffffffff
    for byte in data:
        crc32 = (bit_reverse(byte[]).cast[DType.uint32]() << 24) ^ crc32
        for _ in range(8):
            
            if crc32 & 0x80000000 != 0:
                crc32 = (crc32 << 1) ^ 0x04c11db7
            else:
                crc32 = crc32 << 1
                
    return bit_reverse(crc32^0xffffffff)

fn CRC32_inv(data: List[UInt8], dummy_table: List[UInt32]) -> UInt32:
    """Big endian CRC-32 check using 0xedb88320 as polynomial."""
    var crc32: UInt32 = 0xffffffff

    for byte in data:
        crc32 = (byte[].cast[DType.uint32]() ) ^ crc32

        for _ in range(8):
            if crc32 & 1 != 0:
                crc32 = (crc32 >> 1) ^ 0xedb88320
            else:
                crc32 = crc32 >> 1

    return ~crc32


fn fill_table_n_byte[n: Int]() -> List[UInt32]:
    var table = List[UInt32](capacity=256 * n)
    table.size = 256*n

    for i in range(256 * n):
        if i < 256:
            var key = UInt8(i)
            var crc32 = key.cast[DType.uint32]()
            for _ in range(8):
                if crc32 & 1 != 0:
                    crc32 = (crc32 >> 1) ^ 0xEDB88320
                else:
                    crc32 = crc32 >> 1

            table[i] = crc32
        else:
            var crc32 = table[i - 256]
            var index = int(crc32.cast[DType.uint8]())
            table[i] = (crc32 >> 8) ^ table[index]

    return table

fn CRC32_table_n_byte_compact[size: Int](data: List[UInt8], table: List[UInt32]) -> UInt32:
    var crc32: UInt32 = 0xFFFFFFFF

    alias step_size = 4 # Always smashing 4 bytes into an UInt32
    alias units = size // step_size

    var length = len(data) // size
    var extra = len(data) % size

    var n = 0

    for i in range(start=0, end=length * size, step=size):
        @parameter
        for j in range(units):
            var vals = (
                (data[i + j * step_size + 3].cast[DType.uint32]() << 24)
                | (data[i + j * step_size + 2].cast[DType.uint32]() << 16)
                | (data[i + j * step_size + 1].cast[DType.uint32]() << 8)
                | (data[i + j * step_size + 0].cast[DType.uint32]() << 0)
            )

            if j == 0:
                vals = vals ^ crc32
                crc32 = 0 
        

            n = size - j * step_size
            crc32 = (
                table[(n - 4) * 256 + int((vals >> 24).cast[DType.uint8]())]
                ^ table[(n - 3) * 256 + int((vals >> 16).cast[DType.uint8]())]
                ^ table[(n - 2) * 256 + int((vals >> 8).cast[DType.uint8]())]
                ^ table[(n - 1) * 256 + int((vals >> 0).cast[DType.uint8]())]     
            ) ^ crc32
            

    for i in range(size * length, size * length + extra):
        var index = (crc32 ^ data[i].cast[DType.uint32]()) & 0xFF
        crc32 = table[int(index)] ^ (crc32 >> 8)

    return crc32 ^ 0xFFFFFFFF


alias data_table_func = fn(data: List[UInt8], table: List[UInt32]) -> UInt32


fn wrap_func[testing_function: data_table_func, data: List[UInt8], table: List[UInt32] ]():
    var a =  testing_function(data = data, table = table)
    benchmark.keep(a)


fn bench[function_1: data_table_func, function_2: data_table_func,
         test_list: List[UInt8], test_table: List[UInt32]]() raises -> Tuple[Float64, Float64]:


    var CRC32_1 = function_1(test_list, test_table) 

    alias function_1_wrapped = wrap_func[function_1]
    var report_1 = benchmark.run[function_1_wrapped[test_list, test_table]](max_runtime_secs=1.0
    ).mean(benchmark.Unit.ms)
    print("Function 1 runtime (ms): \t", report_1)


    var CRC32_2 = function_2(test_list, test_table) 
    
    #assert_equal(CRC32_1, CRC32_2)

    alias function_2_wrapped = wrap_func[function_2]
    var report_2 = benchmark.run[function_2_wrapped[test_list, test_table]](max_runtime_secs=1.0
    ).mean(benchmark.Unit.ms)
    print("Function 2 runtime (ms): \t", report_2)

    print("Speedup factor: \t\t", (report_1/report_2))

    return report_1, report_2


def main():


    alias fill_size = 2**20
    alias g = UnsafePointer[UInt8].alloc(fill_size)
    rand[DType.uint8](ptr =  g, size = fill_size)
    alias rand_list = List[UInt8](unsafe_pointer = g, size = fill_size, capacity = fill_size)

    @parameter
    for i in range(4, 68, 4):
        print(i)
        _, _ = bench[CRC32, CRC32_table_n_byte_compact[i], rand_list, fill_table_n_byte[i]()]()