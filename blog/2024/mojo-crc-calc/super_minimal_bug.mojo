fn fill_super_minimal() -> List[UInt32]:
    var table = List[UInt32](capacity=5)


    table.append(129)

    for i in range(5):
        table.append(int(table[i].cast[DType.uint8]()))

    return table

def main():

    var var_table = fill_super_minimal()
    alias alias_table = fill_super_minimal()

    for i in range(5):
        print(var_table[i], alias_table[i])
