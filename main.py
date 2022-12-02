from fastapi import FastAPI, Request, Form, File, UploadFile
from pydantic import BaseModel
from typing import List
import re
import numpy as np
import pandas as pd
from typing import Union
from pickle import load
#from fastapi.templating import Jinja2Templates
#from fastapi.staticfiles import StaticFiles


#templates = Jinja2Templates(directory="templates")

app = FastAPI()

# так вводятся данные для одного элемента
class Item(BaseModel):
    name: str
    year: int
    selling_price: Union[int, None] = None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

# функция для обработки torque

def for_torque(el):
    try:
        n = re.findall('\d+[,| . ]\d+|\d+', el)
        tor = float(n[0].replace(',', ''))
        max_tor = float(n[-1].replace(',', ''))
        
        if re.findall('nm', el.lower()) != []:
            unit = 'nm'
        elif re.findall('kgm', el.lower()) != []:
            unit = 'kgm'
            tor *= 9.80665
        else:
            unit = 'CHECK'
            
    except:
        tor = np.nan
        max_tor = np.nan
        unit = np.nan
    return tor, max_tor

def to_float(el):
    try:
        if re.findall('\d+\.\d+', el) != []:
            return float(re.findall('\d+\.\d+', el)[0])
        else:
            return float(re.findall('\d+', el)[0])
    except:
        return np.nan  #  операция не сработает только для пропусков

def car_name(el):
    return el.split()[0]

@app.post("/predict_item")
async def predict_item(item: Item) -> float:

    # форматирование признаков
    year = item.year
    selling_price = item.selling_price
    km_driven = item.km_driven
    fuel = item.fuel
    seller_type = item.seller_type
    transmission = item.transmission
    owner = item.owner
    mileage = to_float(item.mileage)
    engine = to_float(item.engine)
    max_power = to_float(item.max_power)
    torque_nm, max_torque_rpm = for_torque(item.torque)
    mark = item.name.split()[0]
    seats = int(item.seats) # int

    # формируем векторы
    X = pd.DataFrame(np.array([year, km_driven, mileage, engine, max_power, seats, torque_nm, max_torque_rpm]).reshape(1,8), columns=['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'torque_nm', 'max_torque_rpm'])
    X_cat = pd.DataFrame(np.array([fuel, seller_type, transmission, owner, seats, mark]).reshape(1,6), columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats', 'mark'])
    print(list(X_cat.to_numpy()))

    # масштабирование
    scaler = load(open('standard_scaler_base.pkl', 'rb'))
    X_scaled = pd.DataFrame(scaler.transform(X))

    # кодирование категориальных переменных
    enc = load(open('one_hot_encoder_with_cars.pkl', 'rb'))
    X_dummies = enc.transform(X_cat).toarray()
    X_data = pd.DataFrame(X_dummies, columns=list(enc.get_feature_names_out()))
    X_data['year'] = X_scaled[0]
    X_data['km_driven'] = X_scaled[1]
    X_data['mileage'] = X_scaled[2]
    X_data['engine'] = X_scaled[3]
    X_data['max_power'] = X_scaled[4]
    X_data['seats'] = X_scaled[5]
    X_data['torque_nm'] = X_scaled[6]
    X_data['max_torque_rpm'] = X_scaled[7]
    X_data.drop('seats', axis=1, inplace=True)
    

    # формируем новые признаки
    X_data['engine_to_capacity'] = X_data['max_power']/X_data['engine'] #  мощность двигателя к объему
    X_data['year2'] = X_data['year']**2 #  посмотрим на квадратичную зависимость между годом и ценой
    X_data['km_driven2'] = X_data['km_driven']**2 #  посмотрим на квадратичную зависимость между годом и km_driven
    X_data['engine2'] = X_data['engine']**2 #  посмотрим на квадратичную зависимость между годом и engine
    X_data['max_power2'] = X_data['max_power']**2 #  посмотрим на квадратичную зависимость между годом и max_power

    # боремся с мультиколлинеарностью
    for col in ['fuel_CNG', 'seller_type_Dealer', 'transmission_Automatic', 'owner_First Owner', 'seats_10', 'mark_Ambassador']:
        X_data.drop(col, axis=1, inplace=True)

    # предсказание
    model = load(open('model_elastic_with_cars.pkl', 'rb'))
    pred = model.predict(X_data)
    item.selling_price = int(pred[0])

    #return X_data
    return {'selling_price': item.selling_price}


def car_name(el):
    return el.split()[0]
def to_int(el):
    return int(el)

@app.post("/predict_items")
async def predict_items(items: List[Item]) -> List[float]:

    # формируем датасет
    X = pd.DataFrame()

    for i in items:
        X = X.append(dict(i), ignore_index=True)

    tor = []
    max_tor = []

    #return {'ans': list(X.columns)}
    
    for i in X['torque']:
        tor_v, max_tor_v = for_torque(i)
        tor.append(tor_v)
        max_tor.append(max_tor_v)
    X['torque_nm'] = tor
    X['max_torque_rpm'] = max_tor

    for feature in ['mileage', 'engine', 'max_power']:
        X[feature] = list(map(to_float, X[feature]))

    X.drop(['torque', 'selling_price'], axis=1, inplace=True)

    X['mark'] = list(map(car_name, X['name']))

    X_cat = pd.DataFrame(X[['fuel', 'seller_type', 'transmission', 'owner', 'seats', 'mark']])
    X = pd.DataFrame(X[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'torque_nm', 'max_torque_rpm']])

    # масштабирование
    scaler = load(open('standard_scaler_base.pkl', 'rb'))
    X_scaled = pd.DataFrame(scaler.transform(X))
    X_cat['seats']= list(map(int, X_cat['seats']))
    X_cat['seats']= list(map(str, X_cat['seats']))

    
    # кодирование категориальных переменных
    enc = load(open('one_hot_encoder_with_cars.pkl', 'rb'))
    
    print(list(X_cat.to_numpy()))
    X_dummies = enc.transform(X_cat.to_numpy()).toarray() #.to_numpy()
    X_data = pd.DataFrame(X_dummies, columns=list(enc.get_feature_names_out()))
    X_data['year'] = X_scaled[0]
    X_data['km_driven'] = X_scaled[1]
    X_data['mileage'] = X_scaled[2]
    X_data['engine'] = X_scaled[3]
    X_data['max_power'] = X_scaled[4]
    X_data['seats'] = X_scaled[5]
    X_data['torque_nm'] = X_scaled[6]
    X_data['max_torque_rpm'] = X_scaled[7]
    X_data.drop('seats', axis=1, inplace=True)


    # формируем новые признаки
    X_data['engine_to_capacity'] = X_data['max_power']/X_data['engine'] #  мощность двигателя к объему
    X_data['year2'] = X_data['year']**2 #  посмотрим на квадратичную зависимость между годом и ценой
    X_data['km_driven2'] = X_data['km_driven']**2 #  посмотрим на квадратичную зависимость между годом и km_driven
    X_data['engine2'] = X_data['engine']**2 #  посмотрим на квадратичную зависимость между годом и engine
    X_data['max_power2'] = X_data['max_power']**2 #  посмотрим на квадратичную зависимость между годом и max_power

    # боремся с мультиколлинеарностью
    for col in ['fuel_CNG', 'seller_type_Dealer', 'transmission_Automatic', 'owner_First Owner', 'seats_10', 'mark_Ambassador']:
        X_data.drop(col, axis=1, inplace=True)

    # предсказание
    model = load(open('model_elastic_with_cars.pkl', 'rb'))
    pred = model.predict(X_data)
    #selling_price = pred

    # оформление результатов
    for i in range(len(items)):
        items[i].selling_price = pred[i]
    

    #return {'result': X_data}
    return items