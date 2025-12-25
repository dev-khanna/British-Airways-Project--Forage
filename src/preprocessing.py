import pandas as pd

def initial_preprocessing(df):
    # Deleting index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # a.) sales_channel [nominal]
    df["sales_channel_internet"]=0
    internet_el=[]
    mobile_el=[]
    df["sales_channel_mobile"]=0

    for el in df["sales_channel"]:
        if el=="Internet":
            internet_el.append(1)
        else:
            internet_el.append(0)

        if el=="Mobile":
            mobile_el.append(1)
        else:
            mobile_el.append(0)
    df["sales_channel_internet"]=internet_el
    df["sales_channel_mobile"]=mobile_el
    df=df.drop(columns=["sales_channel"])

    # b.) trip_type [nominal]
    df["trip_type_round"]=0
    df["trip_type_circle"]=0
    df["trip_type_oneway"]=0
    trip_type_round_el=[]
    trip_type_circle_el=[]
    trip_type_one_el=[]

    for el in df["trip_type"]:
        if el=="RoundTrip":
            trip_type_round_el.append(1)
        else :
            trip_type_round_el.append(0)
        if el=="CircleTrip":
            trip_type_circle_el.append(1)
        else:
            trip_type_circle_el.append(0)
        if el=="OneWay":
            trip_type_one_el.append(1)
        else:
            trip_type_one_el.append(0)

    df["trip_type_round"]=trip_type_round_el
    df["trip_type_circle"]=trip_type_circle_el
    df["trip_type_oneway"]=trip_type_one_el
    df=df.drop(columns=["trip_type"])

    # c.) flight_day [ordinal]
    df["flight_day_num"]=0
    flight_day_el=[]
    for el in df["flight_day"]:
        if el=="Mon":
            flight_day_el.append(1)
        elif el=="Tue":
            flight_day_el.append(2)
        elif el=="Wed":
            flight_day_el.append(3)
        elif el=="Thu":
            flight_day_el.append(4)
        elif el=="Fri":
            flight_day_el.append(5)
        elif el=="Sat":
            flight_day_el.append(6)
        elif el=="Sun":
            flight_day_el.append(7)
        
    df["flight_day_num"]=flight_day_el
    df=df.drop(columns=["flight_day"])
    
    return df

def frequency_encode_column(x_train, x_test, column_name):
    # Your exact hashmap logic
    hash1={}
    l1=[]
    l11=[]

    # x_train operations
    for el in x_train[column_name]:
        if el in hash1:
            hash1[el]+=1
        else:
            hash1[el]=1
    for el in x_train[column_name]:
        l1.append(hash1[el])
    x_train[column_name]=l1

    # x_test operations
    for el in x_test[column_name]:
        if el in hash1:
            l11.append(hash1[el])
        else:
            l11.append(0)
    x_test[column_name]=l11
    
    return x_train, x_test