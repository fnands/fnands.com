fn fill_super_minimal() -> List[UInt32]:
    var table = List[UInt32](capacity=2)

    table.append(129)
    table.append(int(table[0].cast[DType.uint8]()))

    #var a: UInt32 = 129
    #table.append(int(a.cast[DType.uint8]()))



    return table

def main():

    var var_table = fill_super_minimal()
    alias alias_table = fill_super_minimal()

    for i in range(2):
        print(var_table[i], alias_table[i])
