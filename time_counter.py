def time_count_second(day, hour, minute, second):
    print(day*3600*24 + hour*3600 + minute*60 + second)
    return


if __name__ == "__main__":
    print("###########32")
    time_count_second(day=1,hour=2,minute=31,second=46)

    print("###########64")
    time_count_second(day=0,hour=13,minute=19,second=2)

    print("###########128")
    time_count_second(day=0,hour=6,minute=28,second=8)
    
    print("###########256")
    time_count_second(day=0,hour=3,minute=40,second=19)

    print("###########512")
    time_count_second(day=0,hour=2,minute=16,second=38)

    print("###########1024")
    time_count_second(day=0,hour=1,minute=42,second=37)

    print("###########2048")
    time_count_second(day=0,hour=1,minute=29,second=2)

    

