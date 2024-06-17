
fn fill_table_2_byte() -> List[UInt32]:
    var table = List[UInt32](capacity=512)

    #table.size = 1024
    #table.resize(512, 0)
    for i in range(512):

        if i < 256: 
            var key = UInt8(i)
            var crc32 = key.cast[DType.uint32]()
            for _ in range(8):
                if crc32 & 1 != 0:
                    crc32 = (crc32 >> 1) ^ 0xedb88320
                else:
                    crc32 = crc32 >> 1

            table.append(crc32)

    
    #for j in range(256, 512):
        if i >= 256: 
            var crc32 = table[i-256]
            table.append((crc32 >> 8) ^ table[int(crc32.cast[DType.uint8]())])
    return table


fn CRC32_table_1_byte(data: List[UInt8], table: List[UInt32]) -> UInt32:
    var crc32: UInt32 = 0xffffffff
    
    for byte in data:
        var index = (crc32 ^ byte[].cast[DType.uint32]()) & 0xff
        crc32 = table[int(index)] ^ (crc32 >> 8)

    return crc32^0xffffffff


fn CRC32_table_2_byte(data: List[UInt8], table: List[UInt32]) -> UInt32:
    var crc32: UInt32 = 0xffffffff

    var length = len(data)//2
    var extra = len(data) % 2

    for i in range(start = 0, end = length *2 , step = 2):
        
        var val: UInt32 = ((data[i + 1].cast[DType.uint32]() << 8) | data[i].cast[DType.uint32]())
        var index = crc32 ^ val
        crc32 =  table[int((index >> 8).cast[DType.uint8]())] ^ table[256 + int(index.cast[DType.uint8]())] ^ (crc32 >> 16)

    for i in range(2*length, 2*length + extra ):
        var index = (crc32 ^ data[i].cast[DType.uint32]()) & 0xff
        crc32 = table[int(index)] ^ (crc32 >> 8)


    return crc32^0xffffffff



fn CRC32(data: List[UInt8], dummy_table: List[UInt32]) -> UInt32:
    """Big endian CRC-32 check using 0xedb88320 as polynomial."""

    var crc32: UInt32 = 0xffffffff

    for byte in data:

        crc32 = (byte[].cast[DType.uint32]() ) ^ crc32

        for _ in range(8):
            if crc32 & 1 != 0:
                crc32 = (crc32 >> 1) ^ 0xedb88320
            else:
                crc32 = crc32 >> 1

    return crc32^0xffffffff

def main():

    var example_list = List[UInt8](5, 78, 138, 1, 54, 172, 104, 99, 54, 167, 94, 56, 22, 184, 204, 90, 201,
                                            5, 78, 138, 1, 54, 172, 104, 99, 54, 167, 94, 56, 22, 184, 204, 90, 201,
                                            5, 78, 138, 1, 54, 172, 104, 99, 54, 167, 94, 56, 22, 184, 204, 90, 201,
                                            5, 78, 138, 1, 54, 172, 104, 99, 54, 167, 94, 56, 22, 184, 204, 90, 201, 
                                            5, 78, 138, 1, 54, 172, 104, 99, 54, 167, 94, 56, 22, 184, 204, 90, 201,
                                            5, 78, 138, 1, 54, 172, 104, 99, 54, 167, 94, 56, 22, 184, 204, 90, 201,
                                            5, 78, 138, 1, 54, 172, 104, 99, 54, 167, 94, 56, 22, 184, 204, 90, 201,
                                            5, 78, 138, 1, 54, 172, 104, 99, 54, 167, 94, 56, 22, 184, 204, 90, 201, 42)


    var var_little_endian_table_2_byte = fill_table_2_byte()
    alias alias_little_endian_table_2_byte = fill_table_2_byte()
    #var mat_little_endian_table_2_byte = alias_little_endian_table_2_byte

    #print(len(var_little_endian_table_2_byte))
    #print(len(alias_little_endian_table_2_byte))

    print(hex(CRC32(example_list, List[UInt32](1))))
    print(hex(CRC32_table_1_byte(example_list, var_little_endian_table_2_byte)))       
    print(hex(CRC32_table_1_byte(example_list, alias_little_endian_table_2_byte)))       

    print(hex(CRC32_table_2_byte(example_list, var_little_endian_table_2_byte)))
    print(hex(CRC32_table_2_byte(example_list, alias_little_endian_table_2_byte)))

    for i in range(0, 256):
        print(var_little_endian_table_2_byte[i], alias_little_endian_table_2_byte[i])
    print("    ")
    for i in range(256, 512):
        print(var_little_endian_table_2_byte[i], alias_little_endian_table_2_byte[i])

    #print(var_little_endian_table_2_byte.unsafe_ptr().is_aligned[64]())
