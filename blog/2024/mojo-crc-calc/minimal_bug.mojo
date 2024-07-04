

fn minimal_table() -> List[UInt32]:
    var table = List[UInt32](capacity=512)

    #table.size = 1024
    #table.resize(512, 0)
    for i in range(512):

        if i < 256: 
            var key = UInt8(i)
            var crc32 = key.cast[DType.uint32]()
            table.append(crc32)

    
        if i >= 256: 
            var crc32 = table[i-256]
            var idx = int(crc32.cast[DType.uint8]())
            table.append(table[idx])
    return table

fn minimal_table_int8() -> List[UInt32]:
    var table = List[UInt32](capacity=512)

    #table.size = 1024
    #table.resize(512, 0)
    for i in range(512):

        if i < 256: 
            var key = UInt8(i)
            var crc32 = key.cast[DType.uint32]()
            table.append(crc32)

    
        if i >= 256: 
            var crc32 = table[i-256]
            var idx = int(crc32.cast[DType.int8]())
            table.append(table[idx])
    return table


fn fill_super_minimal() -> List[UInt32]:
    var table = List[UInt32](capacity=5)


    table.append(129)

    for i in range(5):
        table.append(int(table[i].cast[DType.uint8]()))

    return table

def main():
    var var_min_table = minimal_table()
    alias alias_min_table = minimal_table()

    var var_min_table_unsigned = minimal_table_int8()


    for i in range(0, 512):
        print( var_min_table[i], alias_min_table[i], var_min_table_unsigned[i])

    #var a = fill_super_minimal()
    #alias b = fill_super_minimal()

    #for i in range(5):
    #    print(a[i], b[i])
