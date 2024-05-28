import benchmark
from random import rand, seed
from bit import bit_reverse, byte_reverse
from testing import assert_equal

fn fill_table_n_byte[n: Int]() -> List[UInt32]:
    var table = List[UInt32](capacity=256 * n)
    table.size = 256

    for i in range(256 * n):
        if i < 256:
            var key = UInt8(i)
            var crc32 = key.cast[DType.uint32]()
            for i in range(8):
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


fn CRC32(owned data: List[SIMD[DType.uint8, 1]]) -> SIMD[DType.uint32, 1]:
    var crc32: UInt32 = 0xffffffff
    for byte in data:
        crc32 = (bit_reverse(byte[]).cast[DType.uint32]() << 24) ^ crc32
        for i in range(8):
            
            if crc32 & 0x80000000 != 0:
                crc32 = (crc32 << 1) ^ 0x04c11db7
            else:
                crc32 = crc32 << 1

    return bit_reverse(crc32^0xffffffff)

fn CRC32_table_8_byte(owned data: List[SIMD[DType.uint8, 1]], table: List[UInt32]) -> SIMD[DType.uint32, 1]:
    var crc32: UInt32 = 0xFFFFFFFF
    var size = 8
    var length = len(data) // size
    var extra = len(data) % size

    for i in range(start=0, end=length * size, step=size):
        var val_1: UInt32 = (data[i + 3].cast[DType.uint32]() << 24) 
                          | (data[i + 2].cast[DType.uint32]() << 16) 
                          | (data[i + 1].cast[DType.uint32]() << 8)
                          | (data[i + 0].cast[DType.uint32]() << 0)

        var val_2: UInt32 = (data[i + 7].cast[DType.uint32]() << 24) 
                          | (data[i + 6].cast[DType.uint32]() << 16) 
                          | (data[i + 5].cast[DType.uint32]() << 8) 
                          | (data[i + 4].cast[DType.uint32]() << 0)

        var index_1 = crc32 ^ val_1
        var index_2 = val_2
        crc32 = (
            table[4 * 256 + int((index_1 >> 24).cast[DType.uint8]())]
            ^ table[5 * 256 + int((index_1 >> 16).cast[DType.uint8]())]
            ^ table[6 * 256 + int((index_1 >> 8).cast[DType.uint8]())]
            ^ table[7 * 256 + int((index_1 >> 0).cast[DType.uint8]())]
            ^ table[0 * 256 + int((index_2 >> 24).cast[DType.uint8]())]
            ^ table[1 * 256 + int((index_2 >> 16).cast[DType.uint8]())]
            ^ table[2 * 256 + int((index_2 >> 8).cast[DType.uint8]())]
            ^ table[3 * 256 + int((index_2 >> 0).cast[DType.uint8]())]
        )

    for i in range(size * length, size * length + extra):
        var index = (crc32 ^ data[i].cast[DType.uint32]()) & 0xFF
        crc32 = table[int(index)] ^ (crc32 >> 8)

    return crc32 ^ 0xFFFFFFFF


fn CRC32_table_16_byte(owned data: List[SIMD[DType.uint8, 1]], table: List[UInt32]) -> SIMD[DType.uint32, 1]:
    var crc32: UInt32 = 0xffffffff

    var size = 16

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

        var val_3: UInt32 = (data[i + 11].cast[DType.uint32]() << 24) | 
                            (data[i + 10].cast[DType.uint32]() << 16) | 
                            (data[i + 9].cast[DType.uint32]() << 8) | 
                             data[i + 8].cast[DType.uint32]()

        var val_4: UInt32 = (data[i + 15].cast[DType.uint32]() << 24) | 
                            (data[i + 14].cast[DType.uint32]() << 16) | 
                            (data[i + 13].cast[DType.uint32]() << 8) | 
                             data[i + 12].cast[DType.uint32]()

        var index_1 = crc32 ^ val_1
        var index_2 = val_2
        var index_3 = val_3
        var index_4 = val_4

        crc32 = table[0*256 + int((index_4 >> 24).cast[DType.uint8]())] ^
                table[1*256 + int((index_4 >> 16).cast[DType.uint8]())] ^
                table[2*256 + int((index_4 >> 8).cast[DType.uint8]())] ^
                table[3*256 + int((index_4 >> 0).cast[DType.uint8]())] ^
                table[4*256 + int((index_3 >> 24).cast[DType.uint8]())] ^
                table[5*256 + int((index_3 >> 16).cast[DType.uint8]())] ^
                table[6*256 + int((index_3 >> 8).cast[DType.uint8]())] ^
                table[7*256 + int((index_3 >> 0).cast[DType.uint8]())] ^
                table[8*256 + int((index_2 >> 24).cast[DType.uint8]())] ^
                table[9*256 + int((index_2 >> 16).cast[DType.uint8]())] ^
                table[10*256 + int((index_2 >> 8).cast[DType.uint8]())] ^
                table[11*256 + int((index_2 >> 0).cast[DType.uint8]())] ^
                table[12*256 + int((index_1 >> 24).cast[DType.uint8]())] ^
                table[13*256 + int((index_1 >> 16).cast[DType.uint8]())] ^
                table[14*256 + int((index_1 >> 8).cast[DType.uint8]())] ^
                table[15*256 + int((index_1 >> 0).cast[DType.uint8]())] 
    
    for i in range(size*length, size*length + extra ):
        var index = (crc32 ^ data[i].cast[DType.uint32]()) & 0xff
        crc32 = table[int(index)] ^ (crc32 >> 8)


    return crc32^0xffffffff


fn CRC32_table_32_byte(owned data: List[SIMD[DType.uint8, 1]], table: List[UInt32]) -> SIMD[DType.uint32, 1]:
    var crc32: UInt32 = 0xffffffff

    var size = 32

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

        var val_3: UInt32 = (data[i + 11].cast[DType.uint32]() << 24) | 
                            (data[i + 10].cast[DType.uint32]() << 16) | 
                            (data[i + 9].cast[DType.uint32]() << 8) | 
                             data[i + 8].cast[DType.uint32]()

        var val_4: UInt32 = (data[i + 15].cast[DType.uint32]() << 24) | 
                            (data[i + 14].cast[DType.uint32]() << 16) | 
                            (data[i + 13].cast[DType.uint32]() << 8) | 
                             data[i + 12].cast[DType.uint32]()

        var val_5: UInt32 = (data[i + 19].cast[DType.uint32]() << 24) | 
                            (data[i + 18].cast[DType.uint32]() << 16) | 
                            (data[i + 17].cast[DType.uint32]() << 8) | 
                             data[i + 16].cast[DType.uint32]()

        var val_6: UInt32 = (data[i + 23].cast[DType.uint32]() << 24) | 
                            (data[i + 22].cast[DType.uint32]() << 16) | 
                            (data[i + 21].cast[DType.uint32]() << 8) | 
                             data[i + 20].cast[DType.uint32]()

        var val_7: UInt32 = (data[i + 27].cast[DType.uint32]() << 24) | 
                            (data[i + 26].cast[DType.uint32]() << 16) | 
                            (data[i + 25].cast[DType.uint32]() << 8) | 
                             data[i + 24].cast[DType.uint32]()

        var val_8: UInt32 = (data[i + 31].cast[DType.uint32]() << 24) | 
                            (data[i + 30].cast[DType.uint32]() << 16) | 
                            (data[i + 29].cast[DType.uint32]() << 8) | 
                             data[i + 28].cast[DType.uint32]()

        var index_1 = crc32 ^ val_1.cast[DType.uint32]()
        var index_2 = val_2.cast[DType.uint32]()
        var index_3 = val_3.cast[DType.uint32]()
        var index_4 = val_4.cast[DType.uint32]()
        var index_5 = val_5.cast[DType.uint32]()
        var index_6 = val_6.cast[DType.uint32]()
        var index_7 = val_7.cast[DType.uint32]()
        var index_8 = val_8.cast[DType.uint32]()

        crc32 = table[0*256 + int((index_8 >> 24).cast[DType.uint8]())] ^
                table[1*256 + int((index_8 >> 16).cast[DType.uint8]())] ^
                table[2*256 + int((index_8 >> 8).cast[DType.uint8]())] ^
                table[3*256 + int((index_8 >> 0).cast[DType.uint8]())] ^
                table[4*256 + int((index_7 >> 24).cast[DType.uint8]())] ^
                table[5*256 + int((index_7 >> 16).cast[DType.uint8]())] ^
                table[6*256 + int((index_7 >> 8).cast[DType.uint8]())] ^
                table[7*256 + int((index_7 >> 0).cast[DType.uint8]())] ^
                table[8*256 + int((index_6 >> 24).cast[DType.uint8]())] ^
                table[9*256 + int((index_6 >> 16).cast[DType.uint8]())] ^
                table[10*256 + int((index_6 >> 8).cast[DType.uint8]())] ^
                table[11*256 + int((index_6 >> 0).cast[DType.uint8]())] ^
                table[12*256 + int((index_5 >> 24).cast[DType.uint8]())] ^
                table[13*256 + int((index_5 >> 16).cast[DType.uint8]())] ^
                table[14*256 + int((index_5 >> 8).cast[DType.uint8]())] ^
                table[15*256 + int((index_5 >> 0).cast[DType.uint8]())] ^
                table[16*256 + int((index_4 >> 24).cast[DType.uint8]())] ^
                table[17*256 + int((index_4 >> 16).cast[DType.uint8]())] ^
                table[18*256 + int((index_4 >> 8).cast[DType.uint8]())] ^
                table[19*256 + int((index_4 >> 0).cast[DType.uint8]())] ^
                table[20*256 + int((index_3 >> 24).cast[DType.uint8]())] ^
                table[21*256 + int((index_3 >> 16).cast[DType.uint8]())] ^
                table[22*256 + int((index_3 >> 8).cast[DType.uint8]())] ^
                table[23*256 + int((index_3 >> 0).cast[DType.uint8]())] ^
                table[24*256 + int((index_2 >> 24).cast[DType.uint8]())] ^
                table[25*256 + int((index_2 >> 16).cast[DType.uint8]())] ^
                table[26*256 + int((index_2 >> 8).cast[DType.uint8]())] ^
                table[27*256 + int((index_2 >> 0).cast[DType.uint8]())] ^
                table[28*256 + int((index_1 >> 24).cast[DType.uint8]())] ^
                table[29*256 + int((index_1 >> 16).cast[DType.uint8]())] ^
                table[30*256 + int((index_1 >> 8).cast[DType.uint8]())] ^
                table[31*256 + int((index_1 >> 0).cast[DType.uint8]())] 
    
    for i in range(size*length, size*length + extra ):
        var index = (crc32 ^ data[i].cast[DType.uint32]()) & 0xff
        crc32 = table[int(index)] ^ (crc32 >> 8)


    return crc32^0xffffffff

fn CRC32_table_64_byte(owned data: List[SIMD[DType.uint8, 1]], table: List[UInt32]) -> SIMD[DType.uint32, 1]:
    var crc32: UInt32 = 0xffffffff

    var size = 64

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

        var val_3: UInt32 = (data[i + 11].cast[DType.uint32]() << 24) | 
                            (data[i + 10].cast[DType.uint32]() << 16) | 
                            (data[i + 9].cast[DType.uint32]() << 8) | 
                             data[i + 8].cast[DType.uint32]()

        var val_4: UInt32 = (data[i + 15].cast[DType.uint32]() << 24) | 
                            (data[i + 14].cast[DType.uint32]() << 16) | 
                            (data[i + 13].cast[DType.uint32]() << 8) | 
                             data[i + 12].cast[DType.uint32]()

        var val_5: UInt32 = (data[i + 19].cast[DType.uint32]() << 24) | 
                            (data[i + 18].cast[DType.uint32]() << 16) | 
                            (data[i + 17].cast[DType.uint32]() << 8) | 
                             data[i + 16].cast[DType.uint32]()

        var val_6: UInt32 = (data[i + 23].cast[DType.uint32]() << 24) | 
                            (data[i + 22].cast[DType.uint32]() << 16) | 
                            (data[i + 21].cast[DType.uint32]() << 8) | 
                             data[i + 20].cast[DType.uint32]()

        var val_7: UInt32 = (data[i + 27].cast[DType.uint32]() << 24) | 
                            (data[i + 26].cast[DType.uint32]() << 16) | 
                            (data[i + 25].cast[DType.uint32]() << 8) | 
                             data[i + 24].cast[DType.uint32]()

        var val_8: UInt32 = (data[i + 31].cast[DType.uint32]() << 24) | 
                            (data[i + 30].cast[DType.uint32]() << 16) | 
                            (data[i + 29].cast[DType.uint32]() << 8) | 
                             data[i + 28].cast[DType.uint32]()


        var val_9: UInt32 = (data[i + 35].cast[DType.uint32]() << 24) | 
                            (data[i + 34].cast[DType.uint32]() << 16) | 
                            (data[i + 33].cast[DType.uint32]() << 8) | 
                             data[i + 32].cast[DType.uint32]()

        var val_10: UInt32 =(data[i + 39].cast[DType.uint32]() << 24) | 
                            (data[i + 38].cast[DType.uint32]() << 16) | 
                            (data[i + 37].cast[DType.uint32]() << 8) | 
                             data[i + 36].cast[DType.uint32]()

        var val_11: UInt32 =(data[i + 43].cast[DType.uint32]() << 24) | 
                            (data[i + 42].cast[DType.uint32]() << 16) | 
                            (data[i + 41].cast[DType.uint32]() << 8) | 
                             data[i + 40].cast[DType.uint32]()

        var val_12: UInt32 =(data[i + 47].cast[DType.uint32]() << 24) | 
                            (data[i + 46].cast[DType.uint32]() << 16) | 
                            (data[i + 45].cast[DType.uint32]() << 8) | 
                             data[i + 44].cast[DType.uint32]()

        var val_13: UInt32 =(data[i + 51].cast[DType.uint32]() << 24) | 
                            (data[i + 50].cast[DType.uint32]() << 16) | 
                            (data[i + 49].cast[DType.uint32]() << 8) | 
                             data[i + 48].cast[DType.uint32]()

        var val_14: UInt32 =(data[i + 55].cast[DType.uint32]() << 24) | 
                            (data[i + 54].cast[DType.uint32]() << 16) | 
                            (data[i + 53].cast[DType.uint32]() << 8) | 
                             data[i + 52].cast[DType.uint32]()

        var val_15: UInt32 =(data[i + 59].cast[DType.uint32]() << 24) | 
                            (data[i + 58].cast[DType.uint32]() << 16) | 
                            (data[i + 57].cast[DType.uint32]() << 8) | 
                             data[i + 56].cast[DType.uint32]()

        var val_16: UInt32 =(data[i + 63].cast[DType.uint32]() << 24) | 
                            (data[i + 62].cast[DType.uint32]() << 16) | 
                            (data[i + 61].cast[DType.uint32]() << 8) | 
                             data[i + 60].cast[DType.uint32]()

        var index_1 = crc32 ^ val_1.cast[DType.uint32]()
        var index_2 = val_2.cast[DType.uint32]()
        var index_3 = val_3.cast[DType.uint32]()
        var index_4 = val_4.cast[DType.uint32]()
        var index_5 = val_5.cast[DType.uint32]()
        var index_6 = val_6.cast[DType.uint32]()
        var index_7 = val_7.cast[DType.uint32]()
        var index_8 = val_8.cast[DType.uint32]()
        var index_9 = val_9.cast[DType.uint32]()
        var index_10 = val_10.cast[DType.uint32]()
        var index_11 = val_11.cast[DType.uint32]()
        var index_12 = val_12.cast[DType.uint32]()
        var index_13 = val_13.cast[DType.uint32]()
        var index_14 = val_14.cast[DType.uint32]()
        var index_15 = val_15.cast[DType.uint32]()
        var index_16 = val_16.cast[DType.uint32]()

        crc32 = table[0*256 + int((index_16 >> 24).cast[DType.uint8]())] ^
                table[1*256 + int((index_16 >> 16).cast[DType.uint8]())] ^
                table[2*256 + int((index_16 >> 8).cast[DType.uint8]())] ^
                table[3*256 + int((index_16 >> 0).cast[DType.uint8]())] ^
                table[4*256 + int((index_15 >> 24).cast[DType.uint8]())] ^
                table[5*256 + int((index_15 >> 16).cast[DType.uint8]())] ^
                table[6*256 + int((index_15 >> 8).cast[DType.uint8]())] ^
                table[7*256 + int((index_15 >> 0).cast[DType.uint8]())] ^
                table[8*256 + int((index_14 >> 24).cast[DType.uint8]())] ^
                table[9*256 + int((index_14 >> 16).cast[DType.uint8]())] ^
                table[10*256 + int((index_14 >> 8).cast[DType.uint8]())] ^
                table[11*256 + int((index_14 >> 0).cast[DType.uint8]())] ^
                table[12*256 + int((index_13 >> 24).cast[DType.uint8]())] ^
                table[13*256 + int((index_13 >> 16).cast[DType.uint8]())] ^
                table[14*256 + int((index_13 >> 8).cast[DType.uint8]())] ^
                table[15*256 + int((index_13 >> 0).cast[DType.uint8]())] ^
                table[16*256 + int((index_12 >> 24).cast[DType.uint8]())] ^
                table[17*256 + int((index_12 >> 16).cast[DType.uint8]())] ^
                table[18*256 + int((index_12 >> 8).cast[DType.uint8]())] ^
                table[19*256 + int((index_12 >> 0).cast[DType.uint8]())] ^
                table[20*256 + int((index_11 >> 24).cast[DType.uint8]())] ^
                table[21*256 + int((index_11 >> 16).cast[DType.uint8]())] ^
                table[22*256 + int((index_11 >> 8).cast[DType.uint8]())] ^
                table[23*256 + int((index_11 >> 0).cast[DType.uint8]())] ^
                table[24*256 + int((index_10 >> 24).cast[DType.uint8]())] ^
                table[25*256 + int((index_10 >> 16).cast[DType.uint8]())] ^
                table[26*256 + int((index_10 >> 8).cast[DType.uint8]())] ^
                table[27*256 + int((index_10 >> 0).cast[DType.uint8]())] ^
                table[28*256 + int((index_9 >> 24).cast[DType.uint8]())] ^
                table[29*256 + int((index_9 >> 16).cast[DType.uint8]())] ^
                table[30*256 + int((index_9 >> 8).cast[DType.uint8]())] ^
                table[31*256 + int((index_9 >> 0).cast[DType.uint8]())] ^
                table[32*256 + int((index_8 >> 24).cast[DType.uint8]())] ^
                table[33*256 + int((index_8 >> 16).cast[DType.uint8]())] ^
                table[34*256 + int((index_8 >> 8).cast[DType.uint8]())] ^
                table[35*256 + int((index_8 >> 0).cast[DType.uint8]())] ^
                table[36*256 + int((index_7 >> 24).cast[DType.uint8]())] ^
                table[37*256 + int((index_7 >> 16).cast[DType.uint8]())] ^
                table[38*256 + int((index_7 >> 8).cast[DType.uint8]())] ^
                table[39*256 + int((index_7 >> 0).cast[DType.uint8]())] ^
                table[40*256 + int((index_6 >> 24).cast[DType.uint8]())] ^
                table[41*256 + int((index_6 >> 16).cast[DType.uint8]())] ^
                table[42*256 + int((index_6 >> 8).cast[DType.uint8]())] ^
                table[43*256 + int((index_6 >> 0).cast[DType.uint8]())] ^
                table[44*256 + int((index_5 >> 24).cast[DType.uint8]())] ^
                table[45*256 + int((index_5 >> 16).cast[DType.uint8]())] ^
                table[46*256 + int((index_5 >> 8).cast[DType.uint8]())] ^
                table[47*256 + int((index_5 >> 0).cast[DType.uint8]())] ^
                table[48*256 + int((index_4 >> 24).cast[DType.uint8]())] ^
                table[49*256 + int((index_4 >> 16).cast[DType.uint8]())] ^
                table[50*256 + int((index_4 >> 8).cast[DType.uint8]())] ^
                table[51*256 + int((index_4 >> 0).cast[DType.uint8]())] ^
                table[52*256 + int((index_3 >> 24).cast[DType.uint8]())] ^
                table[53*256 + int((index_3 >> 16).cast[DType.uint8]())] ^
                table[54*256 + int((index_3 >> 8).cast[DType.uint8]())] ^
                table[55*256 + int((index_3 >> 0).cast[DType.uint8]())] ^
                table[56*256 + int((index_2 >> 24).cast[DType.uint8]())] ^
                table[57*256 + int((index_2 >> 16).cast[DType.uint8]())] ^
                table[58*256 + int((index_2 >> 8).cast[DType.uint8]())] ^
                table[59*256 + int((index_2 >> 0).cast[DType.uint8]())] ^
                table[60*256 + int((index_1 >> 24).cast[DType.uint8]())] ^
                table[61*256 + int((index_1 >> 16).cast[DType.uint8]())] ^
                table[62*256 + int((index_1 >> 8).cast[DType.uint8]())] ^
                table[63*256 + int((index_1 >> 0).cast[DType.uint8]())] 
    
    for i in range(size*length, size*length + extra ):
        var index = (crc32 ^ data[i].cast[DType.uint32]()) & 0xff
        crc32 = table[int(index)] ^ (crc32 >> 8)


    return crc32^0xffffffff

from sys.intrinsics import PrefetchLocality, PrefetchRW, PrefetchCache, PrefetchOptions, prefetch

fn CRC32_table_n_byte_compact[
    size: Int
](borrowed data: List[SIMD[DType.uint8, 1]], borrowed table: List[UInt32]) -> SIMD[DType.uint32, 1]:
    var crc32: UInt32 = 0xFFFFFFFF

    alias step_size = 4
    alias units = size // step_size

    var length = len(data) // size
    var extra = len(data) % size

    var vals = List[UInt32](capacity=units)
    vals.size = units
    var interm_crc = List[UInt32](capacity=units)
    interm_crc.size = units
    var n = 0

    #alias prefetch_options = PrefetchOptions().for_read().high_locality().to_data_cache()
    #prefetch[prefetch_options, DType.uint32](table.unsafe_ptr())

    for i in range(start=0, end=length * size, step=size):
        @parameter
        for j in range(units):
            vals[j] = (
                (data[i + j * step_size + 3].cast[DType.uint32]() << 24)
                | (data[i + j * step_size + 2].cast[DType.uint32]() << 16)
                | (data[i + j * step_size + 1].cast[DType.uint32]() << 8)
                | (data[i + j * step_size + 0].cast[DType.uint32]() << 0)
            )

            if j == 0:
                vals[0] = vals[0] ^ crc32

        
        crc32 = 0 
        @parameter
        for j in range(units):
            n = size - j * step_size
            interm_crc[j] = (
                table[(n - 4) * 256 + int((vals[j] >> 24).cast[DType.uint8]())]
                ^ table[(n - 3) * 256 + int((vals[j] >> 16).cast[DType.uint8]())]
                ^ table[(n - 2) * 256 + int((vals[j] >> 8).cast[DType.uint8]())]
                ^ table[(n - 1) * 256 + int((vals[j] >> 0).cast[DType.uint8]())]     
            )
        
        #crc32 = 0
        #@parameter
        #for j in range(units):
            crc32 = crc32^interm_crc[j]

    for i in range(size * length, size * length + extra):
        var index = (crc32 ^ data[i].cast[DType.uint32]()) & 0xFF
        crc32 = table[int(index)] ^ (crc32 >> 8)

    return crc32 ^ 0xFFFFFFFF



fn CRC32_table_8_byte2(owned data: List[SIMD[DType.uint8, 1]], table: List[UInt32]) -> SIMD[DType.uint32, 1]:
    var crc32: UInt32 = 0xFFFFFFFF
    var size = 8
    var bytes_count = len(data)
    var length = bytes_count // size
    var extra = bytes_count % size
    
    var data_pointer = DTypePointer(data.steal_data())
    var data32_pointer = data_pointer.bitcast[DType.uint32]()

    for i in range(start=0, end=length * size, step=size):
        var val = data32_pointer.load[width=2](i//4)

            
        val[0] = crc32 ^ val[0]
        var index: SIMD[DType.uint8, 8]

        index = bitcast[DType.uint8, 8](byte_reverse(val))

        crc32 = 0
        @parameter
        for i in range(8):
            crc32 ^= table[((i + 4) % 8) * 256 + int(index[i])]

    for i in range(size * length, size * length + extra):
        var index = (crc32.cast[DType.uint8]() ^ data[i])
        crc32 = table[int(index)] ^ (crc32 >> 8)

    return crc32 ^ 0xFFFFFFFF

fn run_32[data: List[SIMD[DType.uint8, 1]] ]():
    var a =  CRC32(data)
    benchmark.keep(a)


fn run_32_table_8_byte[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_8_byte(data, table)
    benchmark.keep(a)


fn run_32_table_8_byte_compact[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_n_byte_compact[8](data, table)
    benchmark.keep(a)

fn run_32_table_8_byte_2[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_8_byte2(data, table)
    benchmark.keep(a)


fn run_32_table_16_byte[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_16_byte(data, table)
    benchmark.keep(a)

fn run_32_table_16_byte_compact[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_n_byte_compact[16](data, table)
    benchmark.keep(a)

fn run_32_table_32_byte[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_32_byte(data, table)
    benchmark.keep(a)

fn run_32_table_32_byte_compact[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_n_byte_compact[32](data, table)
    benchmark.keep(a)

fn run_32_table_64_byte[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_64_byte(data, table)
    benchmark.keep(a)

fn run_32_table_64_byte_compact[data: List[SIMD[DType.uint8, 1]], table: List[UInt32]]():
    var a = CRC32_table_n_byte_compact[64](data, table)
    benchmark.keep(a)

fn bench() raises:
    alias fill_size = 2**20
    alias g = UnsafePointer[SIMD[DType.uint8, 1]].alloc(fill_size)
    rand[DType.uint8](ptr=g, size=fill_size)

    #alias rand_list = List[SIMD[DType.uint8, 1]](data=g, size=fill_size, capacity=fill_size)
    alias rand_list = List[SIMD[DType.uint8, 1]](unsafe_pointer=g, size=fill_size, capacity=fill_size)

    print( len(rand_list))

    var baseline_crc = CRC32(rand_list)

    var report = benchmark.run[run_32[rand_list]](max_runtime_secs=0.5
    ).mean(benchmark.Unit.ms)
    print("Baseline: \t", report)

    alias little_endian_table_8_byte = fill_table_n_byte[8]()

    assert_equal(baseline_crc, CRC32_table_8_byte(rand_list, little_endian_table_8_byte))

    var report_8 = benchmark.run[run_32_table_8_byte[rand_list, little_endian_table_8_byte]](max_runtime_secs=0.5).mean(
        benchmark.Unit.ms
    )
    print("8 Byte: \t", report_8)

    assert_equal(baseline_crc, CRC32_table_n_byte_compact[8](rand_list, little_endian_table_8_byte))

    var report_8c = benchmark.run[run_32_table_8_byte_compact[rand_list, little_endian_table_8_byte]](
        max_runtime_secs=0.5
    ).mean(benchmark.Unit.ms)
    print("8 Byte (c): \t", report_8c)


    assert_equal(baseline_crc, CRC32_table_8_byte2(rand_list, little_endian_table_8_byte))

    var report_8_2 = benchmark.run[run_32_table_8_byte_2[rand_list, little_endian_table_8_byte]](
        max_runtime_secs=0.5
    ).mean(benchmark.Unit.ms)
    print("8 Byte 2: \t", report_8_2)

    alias little_endian_table_16_byte = fill_table_n_byte[16]()


    assert_equal(baseline_crc, CRC32_table_16_byte(rand_list, little_endian_table_16_byte))

    var report_16 = benchmark.run[run_32_table_16_byte[rand_list, little_endian_table_16_byte]](max_runtime_secs=0.5).mean(
        benchmark.Unit.ms
    )
    print("16 Byte: \t", report_16)

    assert_equal(baseline_crc, CRC32_table_n_byte_compact[16](rand_list, little_endian_table_16_byte))


    var report_16c = benchmark.run[run_32_table_16_byte_compact[rand_list, little_endian_table_16_byte]](
        max_runtime_secs=0.5
    ).mean(benchmark.Unit.ms)
    print("16 Byte (c): \t", report_16c)


    alias little_endian_table_32_byte = fill_table_n_byte[32]()

    assert_equal(baseline_crc, CRC32_table_32_byte(rand_list, little_endian_table_32_byte))


    var report_32 = benchmark.run[run_32_table_32_byte[rand_list, little_endian_table_32_byte]](
        max_runtime_secs=0.5
    ).mean(benchmark.Unit.ms)
    print("32 Byte: \t", report_32)

    assert_equal(baseline_crc, CRC32_table_n_byte_compact[32](rand_list, little_endian_table_32_byte))


    var report_32c = benchmark.run[run_32_table_32_byte_compact[rand_list, little_endian_table_32_byte]](
        max_runtime_secs=0.5
    ).mean(benchmark.Unit.ms)
    print("32 Byte (c): \t", report_32c)

    alias little_endian_table_64_byte = fill_table_n_byte[64]()

    assert_equal(baseline_crc, CRC32_table_64_byte(rand_list, little_endian_table_64_byte))


    var report_64 = benchmark.run[run_32_table_64_byte[rand_list, little_endian_table_64_byte]](
        max_runtime_secs=0.5
    ).mean(benchmark.Unit.ms)
    print("64 Byte: \t", report_64)

    assert_equal(baseline_crc, CRC32_table_n_byte_compact[64](rand_list, little_endian_table_64_byte))


    var report_64c = benchmark.run[run_32_table_64_byte_compact[rand_list, little_endian_table_64_byte]](
        max_runtime_secs=0.5
    ).mean(benchmark.Unit.ms)
    print("64 Byte (c): \t", report_64c)


    print("Speedup 8 Byte: \t",  int(round(report/report_8)))
    print("Speedup 8 Byte (c): \t",  int(round(report/report_8c)))
    print("Speedup 8 2 Byte: \t",  int(round(report/report_8_2)))
    print("Speedup 16 Byte: \t",  int(round(report/report_16)))
    print("Speedup 16 Byte (c): \t",  int(round(report/report_16c)))
    print("Speedup 32 Byte: \t",  int(round(report/report_32 )))
    print("Speedup 32 Byte (c): \t",  int(round(report/report_32c)))
    print("Speedup 64 Byte: \t",  int(round(report/report_64)))
    print("Speedup 64 Byte (c): \t",  int(round(report/report_64c)))


fn main() raises:
    seed(614114419)
    bench()
