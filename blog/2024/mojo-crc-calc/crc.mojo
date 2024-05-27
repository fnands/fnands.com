import benchmark
from random import rand, seed



fn fill_table_n_byte[n: Int]() -> List[UInt32]:

    var table = List[UInt32](capacity=256*n)
    table.size = 256

    for i in range(256*n):

        if i < 256: 
            var key = UInt8(i)
            var crc32 = key.cast[DType.uint32]()
            for i in range(8):
                if crc32 & 1 != 0:
                    crc32 = (crc32 >> 1) ^ 0xedb88320
                else:
                    crc32 = crc32 >> 1

            table[i] = crc32
        else:
            var crc32 = table[i-256]
            var index = int(crc32.cast[DType.uint8]())
            table[i] = (crc32 >> 8) ^ table[index]
            
    return table


fn CRC32_table_8_byte(owned data: List[SIMD[DType.uint8, 1]], table: List[UInt32]) -> SIMD[DType.uint32, 1]:
    var crc32: UInt32 = 0xffffffff

    var size = 8

    var length = len(data)//size
    var extra = len(data) % size



    for i in range(start = 0, end = length*size, step = size):
        
        var val_1: UInt32 = (data[i + 3].cast[DType.uint32]() << 24) | 
                            (data[i + 2].cast[DType.uint32]() << 16) | 
                            (data[i + 1].cast[DType.uint32]() << 8) | 
                             data[i + 0].cast[DType.uint32]()

        var val_2: UInt32 = (data[i + 7].cast[DType.uint32]() << 24) | 
                            (data[i + 6].cast[DType.uint32]() << 16) | 
                            (data[i + 5].cast[DType.uint32]() << 8) | 
                             data[i + 4].cast[DType.uint32]()

        var index_1 = crc32 ^ val_1#.cast[DType.uint32]()
        var index_2 = val_2#.cast[DType.uint32]()
        crc32 = table[4*256 + int((index_1 >> 24).cast[DType.uint8]())] ^
                table[5*256 + int((index_1 >> 16).cast[DType.uint8]())] ^
                table[6*256 + int((index_1 >> 8).cast[DType.uint8]())] ^
                table[7*256 + int((index_1 >> 0).cast[DType.uint8]())] ^
                table[0*256 + int((index_2 >> 24).cast[DType.uint8]())] ^
                table[1*256 + int((index_2 >> 16).cast[DType.uint8]())] ^
                table[2*256 + int((index_2 >> 8).cast[DType.uint8]())] ^
                table[3*256 + int((index_2 >> 0).cast[DType.uint8]())] 
    
    for i in range(size*length, size*length + extra ):
        var index = (crc32 ^ data[i].cast[DType.uint32]()) & 0xff
        crc32 = table[int(index)] ^ (crc32 >> 8)


    return crc32^0xffffffff

fn CRC32_table_n_byte_compact[size: Int](owned data: List[SIMD[DType.uint8, 1]], table: List[UInt32]) -> SIMD[DType.uint32, 1]:
    var crc32: UInt32 = 0xffffffff

    var step_size = 4 # really just 32/8
    var units = size//step_size

    var length = len(data)//size
    var extra = len(data) % size

    var vals = List[UInt32](capacity=units)
    vals.size = units
    var n = 0
    for i in range(start = 0, end = length*size, step = size):
        
        
        
        for j in range(units):
            vals[j] =   (data[i + j*step_size + 3].cast[DType.uint32]() << 24) | 
                        (data[i + j*step_size + 2].cast[DType.uint32]() << 16) | 
                        (data[i + j*step_size + 1].cast[DType.uint32]() << 8) | 
                        (data[i + j*step_size + 0].cast[DType.uint32]() << 0)

            if j == 0:
                vals[0] = vals[0]^crc32
                crc32 = 0
        #for j in range(units):
            n = size - j*step_size
            crc32 = table[(n-4)*256 + int((vals[j] >> 24).cast[DType.uint8]())] ^
                    table[(n-3)*256 + int((vals[j] >> 16).cast[DType.uint8]())] ^
                    table[(n-2)*256 + int((vals[j] >> 8).cast[DType.uint8]())] ^
                    table[(n-1)*256 + int((vals[j] >> 0).cast[DType.uint8]())] ^ crc32

    
    for i in range(size*length, size*length + extra ):
        var index = (crc32 ^ data[i].cast[DType.uint32]()) & 0xff
        crc32 = table[int(index)] ^ (crc32 >> 8)


    return crc32^0xffffffff



fn run_32_table_8_byte[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_8_byte(data, table)
    benchmark.keep(a)

fn run_32_table_8_byte_compact[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_n_byte_compact[16](data, table)
    benchmark.keep(a)


fn bench():

    
    alias fill_size = 2**20
    alias g = UnsafePointer[SIMD[DType.uint8, 1]].alloc(fill_size)
    rand[DType.uint8](ptr =  g, size = fill_size)


    alias rand_list = List[SIMD[DType.uint8,1]](data = g, size = fill_size, capacity = fill_size)


    print(len(rand_list))


    alias little_endian_table_8_byte = fill_table_n_byte[8]()




    var report_6 = benchmark.run[run_32_table_8_byte[rand_list, little_endian_table_8_byte]](max_runtime_secs=5
    ).mean(benchmark.Unit.ms)
    print(report_6)

    alias little_endian_table_16_byte = fill_table_n_byte[16]()


    
    var report_7 = benchmark.run[run_32_table_8_byte_compact[rand_list, little_endian_table_16_byte]](max_runtime_secs=5
    ).mean(benchmark.Unit.ms)
    print(report_7)



fn main():
    seed(614114419)
    bench()






