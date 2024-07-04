fn fill_super_minimal() -> List[UInt32]:
    var table = List[UInt32](capacity=4)
    table.append(129)

    # This overflows with alias, but not var
    table.append(int(table[0].cast[DType.uint8]()))

    # This overflows in both cases, as if uint8 overflowed. 
    var a: UInt32 = 129
    table.append(int(a.cast[DType.uint8]()))

    # This overflows as if uint32 overflows
    var b: UInt32 = 129
    table.append(UInt32(b.cast[DType.uint8]()))

    return table


#def main():

#    var var_table = fill_super_minimal()
#    alias alias_table = fill_super_minimal()

#    for i in range(4):
#        print(var_table[i], alias_table[i])

    # Extra wtf
#    var b: UInt32 = 129
#    print(int(b.cast[DType.uint8]()))

fn main():

    var b_v: UInt32 = 129
    alias b_a: UInt32 = 129

    var c_v = b_v.cast[DType.uint8]()
    alias c_a = b_a.cast[DType.uint8]()

    print(c_v, c_a)

    var d_v = int(b_v.cast[DType.uint8]())
    alias d_a = int(b_a.cast[DType.uint8]())

    print(d_v, d_a)


    var e_v = UInt32(b_v.cast[DType.uint8]())
    alias e_a = UInt32(b_a.cast[DType.uint8]())

    print(e_v, e_a)

    var a: UInt32 = 129

    print(int(a.cast[DType.uint8]()))
