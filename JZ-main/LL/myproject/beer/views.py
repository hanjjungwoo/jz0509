from django.shortcuts import render, redirect
from django.db import transaction
from django.core.paginator import Paginator

from django.core.serializers.json import DjangoJSONEncoder

from django.http import HttpResponse, request, response

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import sklearn as sk
import warnings
# 직렬화
from rest_framework import viewsets
import csv
import random

from .models import *

from django.conf import settings
from user.models import User

warnings.filterwarnings('ignore')


# 우리가 예측한 평점과 실제 평점간의 차이를 MSE로 계산
def get_mse(pred, actual):
    # 평점이 있는 실제 영화만 추출
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# 특정 도시와 비슷한 유사도를 가지는 도시 Top_N에 대해서만 적용 -> 시간오래걸림
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 도시 개수만큼 루프
    for col in range(ratings_arr.shape[1]):
        # 유사도 행렬에서 유사도가 큰 순으로 n개의 데이터 행렬의 인덱스 반환
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n - 1:-1]]
        # 개인화된 예측 평점 계산 : 각 col 도시별(1개), 2496 사용자들의 예측평점
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(
                ratings_arr[row, :][top_n_items].T)
            pred[row, col] /= np.sum(item_sim_arr[col, :][top_n_items])

    return pred


def get_not_tried_beer(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 도시 정보를 추출해 Series로 반환
    # 반환된 user_rating은 영화명(title)을 인덱스로 가지는 Series 객체
    user_rating = ratings_matrix.loc[userId, :]

    # user_rating이 0보다 크면 기존에 관란함 영화.
    # 대상 인덱스를 추출해 list 객체로 만듦
    tried = user_rating[user_rating > 0].index.tolist()

    # 모든 도시명을 list 객체로 만듦
    beer_list = ratings_matrix.columns.tolist()

    # list comprehension으로 tried에 해당하는 도시는 beer_list에서 제외
    not_tried = [beer for beer in beer_list if beer not in tried]

    return not_tried


# 예측 평점 DataFrame에서 사용자 id 인덱스와 not_tried로 들어온 도시명 추출 후
# 가장 예측 평점이 높은 순으로 정렬


def recomm_beer_by_userid(pred_df, userId, not_tried, top_n):
    recomm_beer = pred_df.loc[userId,
                              not_tried].sort_values(ascending=False)[:top_n]
    return recomm_beer


def recomm_feature(df):

    ratings = df[['장소', '아이디', '평점']]
    # 피벗 테이블을 이용해 유저-아이디 매트릭스 구성
    ratings_matrix = ratings.pivot_table('평점', index='아이디', columns='장소')
    ratings_matrix.head(3)

    # fillna함수를 이용해 Nan처리
    ratings_matrix = ratings_matrix.fillna(0)

    # 유사도 계산을 위해 트랜스포즈
    ratings_matrix_T = ratings_matrix.transpose()

    # 아이템-유저 매트릭스로부터 코사인 유사도 구하기
    item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

    # cosine_similarity()로 반환된 넘파이 행렬에 영화명을 매핑해 DataFrame으로 변환
    item_sim_df = pd.DataFrame(data=item_sim,
                               index=ratings_matrix.columns,
                               columns=ratings_matrix.columns)

    return item_sim_df


def recomm_beer(item_sim_df, beer_name):
    # 해당 도시와 유사도가 높은 도시 5개만 추천
    return item_sim_df[beer_name].sort_values(ascending=False)[1:10]


def recomm_detail(item_sim_df, detail):
    # 해당 도시와 유사도가 높은 도시 5개만 추천
    return item_sim_df[detail].sort_values(ascending=False)[1:10]


def session(request):
    beer_list = pd.rçead_csv('result.csv', encoding='utf-8', index_col=0)
    beer_list = beer_list['place']
    login = {}

    login_session = request.session.get('login_session')

    if login_session == '':
        login['login_session'] = False
    else:
        login['login_session'] = True
    return render(request, 'beer/ver1_result.html', login)


def ver1(request):
    beer_list = pd.read_csv('result.csv', encoding='utf-8', index_col=0)
    ratings = pd.read_csv('merge.csv', encoding='utf-8', index_col=0)
    cluster_3 = pd.read_csv('대표군집클러스터링.csv', encoding='utf-8', index_col=0)
    cluster_all = pd.read_csv('전체도시클러스터링.csv', encoding='utf-8', index_col=0)
    beer_list = beer_list['locate']

    # ver1에서 로그인 세션 유지
    context = {'beer_list': beer_list}
    login_session = request.session.get('login_session')

    if login_session == '':
        context['login_session'] = False
    else:
        context['login_session'] = True

    if request.method == 'POST':

        beer_name = request.POST.get('beer', '')
        page = request.GET.get('page')
        df = recomm_feature(ratings)

        result = recomm_beer(df, beer_name)
        result = result.index.tolist()

        # 결과페이지에서 로그인 세션 유지
        login_session = request.session.get('login_session')

        if login_session == '':
            request.session['login_session'] = False
        else:
            request.session['login_session'] = True

        # 가격 등급 리뷰개수 평점 거리 필터링
        hotel1 = Hotel.objects.filter(place=result[0])
        hotel1_1to10 = hotel1[0:10]
        hotel1_11to20 = hotel1[11:21]
        hotel1_21to30 = hotel1[22:32]
        hotel1_31to40 = hotel1[33:43]
        hotel1_41to50 = hotel1[44:54]
        hotel1_51to60 = hotel1[55:65]
        hotel1_61to70 = hotel1[66:76]
        hotel1_71to80 = hotel1[77:87]
        hotel1_81to90 = hotel1[88:98]
        hotel1_91to100 = hotel1[99:109]
        hotel1_101to110 = hotel1[110:120]
        hotel1_111to120 = hotel1[121:131]
        hotel1_121to130 = hotel1[132:142]
        hotel1_131to140 = hotel1[143:153]
        hotel1_141to150 = hotel1[154:164]
        hotel1_151to160 = hotel1[165:175]
        hotel1_161to170 = hotel1[176:186]
        hotel1_171to180 = hotel1[187:197]
        hotel1_181to190 = hotel1[198:208]
        hotel1_191to200 = hotel1[209:219]
        hotel1_201to210 = hotel1[220:230]
        hotel1_211to220 = hotel1[231:241]

        hotel1_cost_up = hotel1.order_by('cost')
        hotel1_cost_up_1to10 = hotel1_cost_up[0:10]
        hotel1_cost_up_11to20 = hotel1_cost_up[11:21]
        hotel1_cost_up_21to30 = hotel1_cost_up[22:32]
        hotel1_cost_up_31to40 = hotel1_cost_up[33:43]
        hotel1_cost_up_41to50 = hotel1_cost_up[44:54]
        hotel1_cost_up_51to60 = hotel1_cost_up[55:65]
        hotel1_cost_up_61to70 = hotel1_cost_up[66:76]
        hotel1_cost_up_71to80 = hotel1_cost_up[77:87]
        hotel1_cost_up_81to90 = hotel1_cost_up[88:98]
        hotel1_cost_up_91to100 = hotel1_cost_up[99:109]
        hotel1_cost_up_101to110 = hotel1_cost_up[110:120]
        hotel1_cost_up_111to120 = hotel1_cost_up[121:131]
        hotel1_cost_up_121to130 = hotel1_cost_up[132:142]
        hotel1_cost_up_131to140 = hotel1_cost_up[143:153]
        hotel1_cost_up_141to150 = hotel1_cost_up[154:164]
        hotel1_cost_up_151to160 = hotel1_cost_up[165:175]
        hotel1_cost_up_161to170 = hotel1_cost_up[176:186]
        hotel1_cost_up_171to180 = hotel1_cost_up[187:197]
        hotel1_cost_up_181to190 = hotel1_cost_up[198:208]
        hotel1_cost_up_191to200 = hotel1_cost_up[209:219]
        hotel1_cost_up_201to210 = hotel1_cost_up[220:230]
        hotel1_cost_up_211to220 = hotel1_cost_up[231:241]

        hotel1_cost_down = hotel1.order_by('-cost')
        hotel1_cost_down_1to10 = hotel1_cost_down[0:10]
        hotel1_cost_down_11to20 = hotel1_cost_down[11:21]
        hotel1_cost_down_21to30 = hotel1_cost_down[22:32]
        hotel1_cost_down_31to40 = hotel1_cost_down[33:43]
        hotel1_cost_down_41to50 = hotel1_cost_down[44:54]
        hotel1_cost_down_51to60 = hotel1_cost_down[55:65]
        hotel1_cost_down_61to70 = hotel1_cost_down[66:76]
        hotel1_cost_down_71to80 = hotel1_cost_down[77:87]
        hotel1_cost_down_81to90 = hotel1_cost_down[88:98]
        hotel1_cost_down_91to100 = hotel1_cost_down[99:109]
        hotel1_cost_down_101to110 = hotel1_cost_down[110:120]
        hotel1_cost_down_111to120 = hotel1_cost_down[121:131]
        hotel1_cost_down_121to130 = hotel1_cost_down[132:142]
        hotel1_cost_down_131to140 = hotel1_cost_down[143:153]
        hotel1_cost_down_141to150 = hotel1_cost_down[154:164]
        hotel1_cost_down_151to160 = hotel1_cost_down[165:175]
        hotel1_cost_down_161to170 = hotel1_cost_down[176:186]
        hotel1_cost_down_171to180 = hotel1_cost_down[187:197]
        hotel1_cost_down_181to190 = hotel1_cost_down[198:208]
        hotel1_cost_down_191to200 = hotel1_cost_down[209:219]
        hotel1_cost_down_201to210 = hotel1_cost_down[220:230]
        hotel1_cost_down_211to220 = hotel1_cost_down[231:241]

        hotel1_rating_down = hotel1.order_by('-rating')
        hotel1_rating_down_1to10 = hotel1_rating_down[0:10]
        hotel1_rating_down_11to20 = hotel1_rating_down[11:21]
        hotel1_rating_down_21to30 = hotel1_rating_down[22:32]
        hotel1_rating_down_31to40 = hotel1_rating_down[33:43]
        hotel1_rating_down_41to50 = hotel1_rating_down[44:54]
        hotel1_rating_down_51to60 = hotel1_rating_down[55:65]
        hotel1_rating_down_61to70 = hotel1_rating_down[66:76]
        hotel1_rating_down_71to80 = hotel1_rating_down[77:87]
        hotel1_rating_down_81to90 = hotel1_rating_down[88:98]
        hotel1_rating_down_91to100 = hotel1_rating_down[99:109]
        hotel1_rating_down_101to110 = hotel1_rating_down[110:120]
        hotel1_rating_down_111to120 = hotel1_rating_down[121:131]
        hotel1_rating_down_121to130 = hotel1_rating_down[132:142]
        hotel1_rating_down_131to140 = hotel1_rating_down[143:153]
        hotel1_rating_down_141to150 = hotel1_rating_down[154:164]
        hotel1_rating_down_151to160 = hotel1_rating_down[165:175]
        hotel1_rating_down_161to170 = hotel1_rating_down[176:186]
        hotel1_rating_down_171to180 = hotel1_rating_down[187:197]
        hotel1_rating_down_181to190 = hotel1_rating_down[198:208]
        hotel1_rating_down_191to200 = hotel1_rating_down[209:219]
        hotel1_rating_down_201to210 = hotel1_rating_down[220:230]
        hotel1_rating_down_211to220 = hotel1_rating_down[231:241]

        hotel1_distance_up = hotel1.order_by('distance')
        hotel1_distance_up_1to10 = hotel1_distance_up[0:10]
        hotel1_distance_up_11to20 = hotel1_distance_up[11:21]
        hotel1_distance_up_21to30 = hotel1_distance_up[22:32]
        hotel1_distance_up_31to40 = hotel1_distance_up[33:43]
        hotel1_distance_up_41to50 = hotel1_distance_up[44:54]
        hotel1_distance_up_51to60 = hotel1_distance_up[55:65]
        hotel1_distance_up_61to70 = hotel1_distance_up[66:76]
        hotel1_distance_up_71to80 = hotel1_distance_up[77:87]
        hotel1_distance_up_81to90 = hotel1_distance_up[88:98]
        hotel1_distance_up_91to100 = hotel1_distance_up[99:109]
        hotel1_distance_up_101to110 = hotel1_distance_up[110:120]
        hotel1_distance_up_111to120 = hotel1_distance_up[121:131]
        hotel1_distance_up_121to130 = hotel1_distance_up[132:142]
        hotel1_distance_up_131to140 = hotel1_distance_up[143:153]
        hotel1_distance_up_141to150 = hotel1_distance_up[154:164]
        hotel1_distance_up_151to160 = hotel1_distance_up[165:175]
        hotel1_distance_up_161to170 = hotel1_distance_up[176:186]
        hotel1_distance_up_171to180 = hotel1_distance_up[187:197]
        hotel1_distance_up_181to190 = hotel1_distance_up[198:208]
        hotel1_distance_up_191to200 = hotel1_distance_up[209:219]
        hotel1_distance_up_201to210 = hotel1_distance_up[220:230]
        hotel1_distance_up_211to220 = hotel1_distance_up[231:241]

        hotel1_kind_down = hotel1.order_by('-kind')
        hotel1_kind_down_1to10 = hotel1_kind_down[0:10]
        hotel1_kind_down_11to20 = hotel1_kind_down[11:21]
        hotel1_kind_down_21to30 = hotel1_kind_down[22:32]
        hotel1_kind_down_31to40 = hotel1_kind_down[33:43]
        hotel1_kind_down_41to50 = hotel1_kind_down[44:54]
        hotel1_kind_down_51to60 = hotel1_kind_down[55:65]
        hotel1_kind_down_61to70 = hotel1_kind_down[66:76]
        hotel1_kind_down_71to80 = hotel1_kind_down[77:87]
        hotel1_kind_down_81to90 = hotel1_kind_down[88:98]
        hotel1_kind_down_91to100 = hotel1_kind_down[99:109]
        hotel1_kind_down_101to110 = hotel1_kind_down[110:120]
        hotel1_kind_down_111to120 = hotel1_kind_down[121:131]
        hotel1_kind_down_121to130 = hotel1_kind_down[132:142]
        hotel1_kind_down_131to140 = hotel1_kind_down[143:153]
        hotel1_kind_down_141to150 = hotel1_kind_down[154:164]
        hotel1_kind_down_151to160 = hotel1_kind_down[165:175]
        hotel1_kind_down_161to170 = hotel1_kind_down[176:186]
        hotel1_kind_down_171to180 = hotel1_kind_down[187:197]
        hotel1_kind_down_181to190 = hotel1_kind_down[198:208]
        hotel1_kind_down_191to200 = hotel1_kind_down[209:219]
        hotel1_kind_down_201to210 = hotel1_kind_down[220:230]
        hotel1_kind_down_211to220 = hotel1_kind_down[231:241]

        hotel1_clean_down = hotel1.order_by('-clean')
        hotel1_clean_down_1to10 = hotel1_clean_down[0:10]
        hotel1_clean_down_11to20 = hotel1_clean_down[11:21]
        hotel1_clean_down_21to30 = hotel1_clean_down[22:32]
        hotel1_clean_down_31to40 = hotel1_clean_down[33:43]
        hotel1_clean_down_41to50 = hotel1_clean_down[44:54]
        hotel1_clean_down_51to60 = hotel1_clean_down[55:65]
        hotel1_clean_down_61to70 = hotel1_clean_down[66:76]
        hotel1_clean_down_71to80 = hotel1_clean_down[77:87]
        hotel1_clean_down_81to90 = hotel1_clean_down[88:98]
        hotel1_clean_down_91to100 = hotel1_clean_down[99:109]
        hotel1_clean_down_101to110 = hotel1_clean_down[110:120]
        hotel1_clean_down_111to120 = hotel1_clean_down[121:131]
        hotel1_clean_down_121to130 = hotel1_clean_down[132:142]
        hotel1_clean_down_131to140 = hotel1_clean_down[143:153]
        hotel1_clean_down_141to150 = hotel1_clean_down[154:164]
        hotel1_clean_down_151to160 = hotel1_clean_down[165:175]
        hotel1_clean_down_161to170 = hotel1_clean_down[176:186]
        hotel1_clean_down_171to180 = hotel1_clean_down[187:197]
        hotel1_clean_down_181to190 = hotel1_clean_down[198:208]
        hotel1_clean_down_191to200 = hotel1_clean_down[209:219]
        hotel1_clean_down_201to210 = hotel1_clean_down[220:230]
        hotel1_clean_down_211to220 = hotel1_clean_down[231:241]

        hotel1_conv_down = hotel1.order_by('-conv')
        hotel1_conv_down_1to10 = hotel1_conv_down[0:10]
        hotel1_conv_down_11to20 = hotel1_conv_down[11:21]
        hotel1_conv_down_21to30 = hotel1_conv_down[22:32]
        hotel1_conv_down_31to40 = hotel1_conv_down[33:43]
        hotel1_conv_down_41to50 = hotel1_conv_down[44:54]
        hotel1_conv_down_51to60 = hotel1_conv_down[55:65]
        hotel1_conv_down_61to70 = hotel1_conv_down[66:76]
        hotel1_conv_down_71to80 = hotel1_conv_down[77:87]
        hotel1_conv_down_81to90 = hotel1_conv_down[88:98]
        hotel1_conv_down_91to100 = hotel1_conv_down[99:109]
        hotel1_conv_down_101to110 = hotel1_conv_down[110:120]
        hotel1_conv_down_111to120 = hotel1_conv_down[121:131]
        hotel1_conv_down_121to130 = hotel1_conv_down[132:142]
        hotel1_conv_down_131to140 = hotel1_conv_down[143:153]
        hotel1_conv_down_141to150 = hotel1_conv_down[154:164]
        hotel1_conv_down_151to160 = hotel1_conv_down[165:175]
        hotel1_conv_down_161to170 = hotel1_conv_down[176:186]
        hotel1_conv_down_171to180 = hotel1_conv_down[187:197]
        hotel1_conv_down_181to190 = hotel1_conv_down[198:208]
        hotel1_conv_down_191to200 = hotel1_conv_down[209:219]
        hotel1_conv_down_201to210 = hotel1_conv_down[220:230]
        hotel1_conv_down_211to220 = hotel1_conv_down[231:241]

        hotel1_hotel = Hotel.objects.filter(place=result[0],
                                            classfication='호텔')
        hotel1_hotel_1to10 = hotel1_hotel[0:10]
        hotel1_hotel_11to20 = hotel1_hotel[11:21]
        hotel1_hotel_21to30 = hotel1_hotel[22:32]
        hotel1_hotel_31to40 = hotel1_hotel[33:43]
        hotel1_hotel_41to50 = hotel1_hotel[44:54]
        hotel1_hotel_51to60 = hotel1_hotel[55:65]
        hotel1_hotel_61to70 = hotel1_hotel[66:76]
        hotel1_hotel_71to80 = hotel1_hotel[77:87]
        hotel1_hotel_81to90 = hotel1_hotel[88:98]
        hotel1_hotel_91to100 = hotel1_hotel[99:109]
        hotel1_hotel_101to110 = hotel1_hotel[110:120]
        hotel1_hotel_111to120 = hotel1_hotel[121:131]
        hotel1_hotel_121to130 = hotel1_hotel[132:142]
        hotel1_hotel_131to140 = hotel1_hotel[143:153]
        hotel1_hotel_141to150 = hotel1_hotel[154:164]
        hotel1_hotel_151to160 = hotel1_hotel[165:175]
        hotel1_hotel_161to170 = hotel1_hotel[176:186]
        hotel1_hotel_171to180 = hotel1_hotel[187:197]
        hotel1_hotel_181to190 = hotel1_hotel[198:208]
        hotel1_hotel_191to200 = hotel1_hotel[209:219]
        hotel1_hotel_201to210 = hotel1_hotel[220:230]
        hotel1_hotel_211to220 = hotel1_hotel[231:241]

        hotel1_hostel = Hotel.objects.filter(place=result[0],
                                             classfication='호스텔')
        hotel1_hostel_1to10 = hotel1_hostel[0:10]
        hotel1_hostel_11to20 = hotel1_hostel[11:21]
        hotel1_hostel_21to30 = hotel1_hostel[22:32]
        hotel1_hostel_31to40 = hotel1_hostel[33:43]
        hotel1_hostel_41to50 = hotel1_hostel[44:54]
        hotel1_hostel_51to60 = hotel1_hostel[55:65]
        hotel1_hostel_61to70 = hotel1_hostel[66:76]
        hotel1_hostel_71to80 = hotel1_hostel[77:87]
        hotel1_hostel_81to90 = hotel1_hostel[88:98]
        hotel1_hostel_91to100 = hotel1_hostel[99:109]
        hotel1_hostel_101to110 = hotel1_hostel[110:120]
        hotel1_hostel_111to120 = hotel1_hostel[121:131]
        hotel1_hostel_121to130 = hotel1_hostel[132:142]
        hotel1_hostel_131to140 = hotel1_hostel[143:153]
        hotel1_hostel_141to150 = hotel1_hostel[154:164]
        hotel1_hostel_151to160 = hotel1_hostel[165:175]
        hotel1_hostel_161to170 = hotel1_hostel[176:186]
        hotel1_hostel_171to180 = hotel1_hostel[187:197]
        hotel1_hostel_181to190 = hotel1_hostel[198:208]
        hotel1_hostel_191to200 = hotel1_hostel[209:219]
        hotel1_hostel_201to210 = hotel1_hostel[220:230]
        hotel1_hostel_211to220 = hotel1_hostel[231:241]

        hotel1_guest = Hotel.objects.filter(place=result[0],
                                            classfication='게스트하우스')
        hotel1_guest_1to10 = hotel1_guest[0:10]
        hotel1_guest_11to20 = hotel1_guest[11:21]
        hotel1_guest_21to30 = hotel1_guest[22:32]
        hotel1_guest_31to40 = hotel1_guest[33:43]
        hotel1_guest_41to50 = hotel1_guest[44:54]
        hotel1_guest_51to60 = hotel1_guest[55:65]
        hotel1_guest_61to70 = hotel1_guest[66:76]
        hotel1_guest_71to80 = hotel1_guest[77:87]
        hotel1_guest_81to90 = hotel1_guest[88:98]
        hotel1_guest_91to100 = hotel1_guest[99:109]
        hotel1_guest_101to110 = hotel1_guest[110:120]
        hotel1_guest_111to120 = hotel1_guest[121:131]
        hotel1_guest_121to130 = hotel1_guest[132:142]
        hotel1_guest_131to140 = hotel1_guest[143:153]
        hotel1_guest_141to150 = hotel1_guest[154:164]
        hotel1_guest_151to160 = hotel1_guest[165:175]
        hotel1_guest_161to170 = hotel1_guest[176:186]
        hotel1_guest_171to180 = hotel1_guest[187:197]
        hotel1_guest_181to190 = hotel1_guest[198:208]
        hotel1_guest_191to200 = hotel1_guest[209:219]
        hotel1_guest_201to210 = hotel1_guest[220:230]
        hotel1_guest_211to220 = hotel1_guest[231:241]

        hotel1_apartment = Hotel.objects.filter(place=result[0],
                                                classfication='아파트')
        hotel1_apartment_1to10 = hotel1_apartment[0:10]
        hotel1_apartment_11to20 = hotel1_apartment[11:21]
        hotel1_apartment_21to30 = hotel1_apartment[22:32]
        hotel1_apartment_31to40 = hotel1_apartment[33:43]
        hotel1_apartment_41to50 = hotel1_apartment[44:54]
        hotel1_apartment_51to60 = hotel1_apartment[55:65]
        hotel1_apartment_61to70 = hotel1_apartment[66:76]
        hotel1_apartment_71to80 = hotel1_apartment[77:87]
        hotel1_apartment_81to90 = hotel1_apartment[88:98]
        hotel1_apartment_91to100 = hotel1_apartment[99:109]
        hotel1_apartment_101to110 = hotel1_apartment[110:120]
        hotel1_apartment_111to120 = hotel1_apartment[121:131]
        hotel1_apartment_121to130 = hotel1_apartment[132:142]
        hotel1_apartment_131to140 = hotel1_apartment[143:153]
        hotel1_apartment_141to150 = hotel1_apartment[154:164]
        hotel1_apartment_151to160 = hotel1_apartment[165:175]
        hotel1_apartment_161to170 = hotel1_apartment[176:186]
        hotel1_apartment_171to180 = hotel1_apartment[187:197]
        hotel1_apartment_181to190 = hotel1_apartment[198:208]
        hotel1_apartment_191to200 = hotel1_apartment[209:219]
        hotel1_apartment_201to210 = hotel1_apartment[220:230]
        hotel1_apartment_211to220 = hotel1_apartment[231:241]

        hotel1_apartmenthotel = Hotel.objects.filter(place=result[0],
                                                     classfication='아파트호텔')
        hotel1_apartmenthotel_1to10 = hotel1_apartmenthotel[0:10]
        hotel1_apartmenthotel_11to20 = hotel1_apartmenthotel[11:21]
        hotel1_apartmenthotel_21to30 = hotel1_apartmenthotel[22:32]
        hotel1_apartmenthotel_31to40 = hotel1_apartmenthotel[33:43]
        hotel1_apartmenthotel_41to50 = hotel1_apartmenthotel[44:54]
        hotel1_apartmenthotel_51to60 = hotel1_apartmenthotel[55:65]
        hotel1_apartmenthotel_61to70 = hotel1_apartmenthotel[66:76]
        hotel1_apartmenthotel_71to80 = hotel1_apartmenthotel[77:87]
        hotel1_apartmenthotel_81to90 = hotel1_apartmenthotel[88:98]
        hotel1_apartmenthotel_91to100 = hotel1_apartmenthotel[99:109]
        hotel1_apartmenthotel_101to110 = hotel1_apartmenthotel[110:120]
        hotel1_apartmenthotel_111to120 = hotel1_apartmenthotel[121:131]
        hotel1_apartmenthotel_121to130 = hotel1_apartmenthotel[132:142]
        hotel1_apartmenthotel_131to140 = hotel1_apartmenthotel[143:153]
        hotel1_apartmenthotel_141to150 = hotel1_apartmenthotel[154:164]
        hotel1_apartmenthotel_151to160 = hotel1_apartmenthotel[165:175]
        hotel1_apartmenthotel_161to170 = hotel1_apartmenthotel[176:186]
        hotel1_apartmenthotel_171to180 = hotel1_apartmenthotel[187:197]
        hotel1_apartmenthotel_181to190 = hotel1_apartmenthotel[198:208]
        hotel1_apartmenthotel_191to200 = hotel1_apartmenthotel[209:219]
        hotel1_apartmenthotel_201to210 = hotel1_apartmenthotel[220:230]
        hotel1_apartmenthotel_211to220 = hotel1_apartmenthotel[231:241]

        hotel1_motel = Hotel.objects.filter(place=result[0],
                                            classfication='모텔')
        hotel1_motel_1to10 = hotel1_motel[0:10]
        hotel1_motel_11to20 = hotel1_motel[11:21]
        hotel1_motel_21to30 = hotel1_motel[22:32]
        hotel1_motel_31to40 = hotel1_motel[33:43]
        hotel1_motel_41to50 = hotel1_motel[44:54]
        hotel1_motel_51to60 = hotel1_motel[55:65]
        hotel1_motel_61to70 = hotel1_motel[66:76]
        hotel1_motel_71to80 = hotel1_motel[77:87]
        hotel1_motel_81to90 = hotel1_motel[88:98]
        hotel1_motel_91to100 = hotel1_motel[99:109]
        hotel1_motel_101to110 = hotel1_motel[110:120]
        hotel1_motel_111to120 = hotel1_motel[121:131]
        hotel1_motel_121to130 = hotel1_motel[132:142]
        hotel1_motel_131to140 = hotel1_motel[143:153]
        hotel1_motel_141to150 = hotel1_motel[154:164]
        hotel1_motel_151to160 = hotel1_motel[165:175]
        hotel1_motel_161to170 = hotel1_motel[176:186]
        hotel1_motel_171to180 = hotel1_motel[187:197]
        hotel1_motel_181to190 = hotel1_motel[198:208]
        hotel1_motel_191to200 = hotel1_motel[209:219]
        hotel1_motel_201to210 = hotel1_motel[220:230]
        hotel1_motel_211to220 = hotel1_motel[231:241]

        hotel1_pension = Hotel.objects.filter(place=result[0],
                                              classfication='펜션')
        hotel1_pension_1to10 = hotel1_pension[0:10]
        hotel1_pension_11to20 = hotel1_pension[11:21]
        hotel1_pension_21to30 = hotel1_pension[22:32]
        hotel1_pension_31to40 = hotel1_pension[33:43]
        hotel1_pension_41to50 = hotel1_pension[44:54]
        hotel1_pension_51to60 = hotel1_pension[55:65]
        hotel1_pension_61to70 = hotel1_pension[66:76]
        hotel1_pension_71to80 = hotel1_pension[77:87]
        hotel1_pension_81to90 = hotel1_pension[88:98]
        hotel1_pension_91to100 = hotel1_pension[99:109]
        hotel1_pension_101to110 = hotel1_pension[110:120]
        hotel1_pension_111to120 = hotel1_pension[121:131]
        hotel1_pension_121to130 = hotel1_pension[132:142]
        hotel1_pension_131to140 = hotel1_pension[143:153]
        hotel1_pension_141to150 = hotel1_pension[154:164]
        hotel1_pension_151to160 = hotel1_pension[165:175]
        hotel1_pension_161to170 = hotel1_pension[176:186]
        hotel1_pension_171to180 = hotel1_pension[187:197]
        hotel1_pension_181to190 = hotel1_pension[198:208]
        hotel1_pension_191to200 = hotel1_pension[209:219]
        hotel1_pension_201to210 = hotel1_pension[220:230]
        hotel1_pension_211to220 = hotel1_pension[231:241]

        hotel1_resort = Hotel.objects.filter(place=result[0],
                                             classfication='리조트')
        hotel1_resort_1to10 = hotel1_resort[0:10]
        hotel1_resort_11to20 = hotel1_resort[11:21]
        hotel1_resort_21to30 = hotel1_resort[22:32]
        hotel1_resort_31to40 = hotel1_resort[33:43]
        hotel1_resort_41to50 = hotel1_resort[44:54]
        hotel1_resort_51to60 = hotel1_resort[55:65]
        hotel1_resort_61to70 = hotel1_resort[66:76]
        hotel1_resort_71to80 = hotel1_resort[77:87]
        hotel1_resort_81to90 = hotel1_resort[88:98]
        hotel1_resort_91to100 = hotel1_resort[99:109]
        hotel1_resort_101to110 = hotel1_resort[110:120]
        hotel1_resort_111to120 = hotel1_resort[121:131]
        hotel1_resort_121to130 = hotel1_resort[132:142]
        hotel1_resort_131to140 = hotel1_resort[143:153]
        hotel1_resort_141to150 = hotel1_resort[154:164]
        hotel1_resort_151to160 = hotel1_resort[165:175]
        hotel1_resort_161to170 = hotel1_resort[176:186]
        hotel1_resort_171to180 = hotel1_resort[187:197]
        hotel1_resort_181to190 = hotel1_resort[198:208]
        hotel1_resort_191to200 = hotel1_resort[209:219]
        hotel1_resort_201to210 = hotel1_resort[220:230]
        hotel1_resort_211to220 = hotel1_resort[231:241]

        hotel1_badandbreakfast = Hotel.objects.filter(place=result[0],
                                                      classfication='베드앤브렉퍼스트')
        hotel1_badandbreakfast_1to10 = hotel1_badandbreakfast[0:10]
        hotel1_badandbreakfast_11to20 = hotel1_badandbreakfast[11:21]
        hotel1_badandbreakfast_21to30 = hotel1_badandbreakfast[22:32]
        hotel1_badandbreakfast_31to40 = hotel1_badandbreakfast[33:43]
        hotel1_badandbreakfast_41to50 = hotel1_badandbreakfast[44:54]
        hotel1_badandbreakfast_51to60 = hotel1_badandbreakfast[55:65]
        hotel1_badandbreakfast_61to70 = hotel1_badandbreakfast[66:76]
        hotel1_badandbreakfast_71to80 = hotel1_badandbreakfast[77:87]
        hotel1_badandbreakfast_81to90 = hotel1_badandbreakfast[88:98]
        hotel1_badandbreakfast_91to100 = hotel1_badandbreakfast[99:109]
        hotel1_badandbreakfast_101to110 = hotel1_badandbreakfast[110:120]
        hotel1_badandbreakfast_111to120 = hotel1_badandbreakfast[121:131]
        hotel1_badandbreakfast_121to130 = hotel1_badandbreakfast[132:142]
        hotel1_badandbreakfast_131to140 = hotel1_badandbreakfast[143:153]
        hotel1_badandbreakfast_141to150 = hotel1_badandbreakfast[154:164]
        hotel1_badandbreakfast_151to160 = hotel1_badandbreakfast[165:175]
        hotel1_badandbreakfast_161to170 = hotel1_badandbreakfast[176:186]
        hotel1_badandbreakfast_171to180 = hotel1_badandbreakfast[187:197]
        hotel1_badandbreakfast_181to190 = hotel1_badandbreakfast[198:208]
        hotel1_badandbreakfast_191to200 = hotel1_badandbreakfast[209:219]
        hotel1_badandbreakfast_201to210 = hotel1_badandbreakfast[220:230]
        hotel1_badandbreakfast_211to220 = hotel1_badandbreakfast[231:241]

        hotel1_homestay = Hotel.objects.filter(place=result[0],
                                               classfication='홈스테이')
        hotel1_homestay_1to10 = hotel1_homestay[0:10]
        hotel1_homestay_11to20 = hotel1_homestay[11:21]
        hotel1_homestay_21to30 = hotel1_homestay[22:32]
        hotel1_homestay_31to40 = hotel1_homestay[33:43]
        hotel1_homestay_41to50 = hotel1_homestay[44:54]
        hotel1_homestay_51to60 = hotel1_homestay[55:65]
        hotel1_homestay_61to70 = hotel1_homestay[66:76]
        hotel1_homestay_71to80 = hotel1_homestay[77:87]
        hotel1_homestay_81to90 = hotel1_homestay[88:98]
        hotel1_homestay_91to100 = hotel1_homestay[99:109]
        hotel1_homestay_101to110 = hotel1_homestay[110:120]
        hotel1_homestay_111to120 = hotel1_homestay[121:131]
        hotel1_homestay_121to130 = hotel1_homestay[132:142]
        hotel1_homestay_131to140 = hotel1_homestay[143:153]
        hotel1_homestay_141to150 = hotel1_homestay[154:164]
        hotel1_homestay_151to160 = hotel1_homestay[165:175]
        hotel1_homestay_161to170 = hotel1_homestay[176:186]
        hotel1_homestay_171to180 = hotel1_homestay[187:197]
        hotel1_homestay_181to190 = hotel1_homestay[198:208]
        hotel1_homestay_191to200 = hotel1_homestay[209:219]
        hotel1_homestay_201to210 = hotel1_homestay[220:230]
        hotel1_homestay_211to220 = hotel1_homestay[231:241]

        hotel1_lodge = Hotel.objects.filter(place=result[0],
                                            classfication='롯지')
        hotel1_lodge_1to10 = hotel1_lodge[0:10]
        hotel1_lodge_11to20 = hotel1_lodge[11:21]
        hotel1_lodge_21to30 = hotel1_lodge[22:32]
        hotel1_lodge_31to40 = hotel1_lodge[33:43]
        hotel1_lodge_41to50 = hotel1_lodge[44:54]
        hotel1_lodge_51to60 = hotel1_lodge[55:65]
        hotel1_lodge_61to70 = hotel1_lodge[66:76]
        hotel1_lodge_71to80 = hotel1_lodge[77:87]
        hotel1_lodge_81to90 = hotel1_lodge[88:98]
        hotel1_lodge_91to100 = hotel1_lodge[99:109]
        hotel1_lodge_101to110 = hotel1_lodge[110:120]
        hotel1_lodge_111to120 = hotel1_lodge[121:131]
        hotel1_lodge_121to130 = hotel1_lodge[132:142]
        hotel1_lodge_131to140 = hotel1_lodge[143:153]
        hotel1_lodge_141to150 = hotel1_lodge[154:164]
        hotel1_lodge_151to160 = hotel1_lodge[165:175]
        hotel1_lodge_161to170 = hotel1_lodge[176:186]
        hotel1_lodge_171to180 = hotel1_lodge[187:197]
        hotel1_lodge_181to190 = hotel1_lodge[198:208]
        hotel1_lodge_191to200 = hotel1_lodge[209:219]
        hotel1_lodge_201to210 = hotel1_lodge[220:230]
        hotel1_lodge_211to220 = hotel1_lodge[231:241]

        hotel1_countryhouse = Hotel.objects.filter(place=result[0],
                                                   classfication='컨트리하우스')
        hotel1_countryhouse_1to10 = hotel1_countryhouse[0:10]
        hotel1_countryhouse_11to20 = hotel1_countryhouse[11:21]
        hotel1_countryhouse_21to30 = hotel1_countryhouse[22:32]
        hotel1_countryhouse_31to40 = hotel1_countryhouse[33:43]
        hotel1_countryhouse_41to50 = hotel1_countryhouse[44:54]
        hotel1_countryhouse_51to60 = hotel1_countryhouse[55:65]
        hotel1_countryhouse_61to70 = hotel1_countryhouse[66:76]
        hotel1_countryhouse_71to80 = hotel1_countryhouse[77:87]
        hotel1_countryhouse_81to90 = hotel1_countryhouse[88:98]
        hotel1_countryhouse_91to100 = hotel1_countryhouse[99:109]
        hotel1_countryhouse_101to110 = hotel1_countryhouse[110:120]
        hotel1_countryhouse_111to120 = hotel1_countryhouse[121:131]
        hotel1_countryhouse_121to130 = hotel1_countryhouse[132:142]
        hotel1_countryhouse_131to140 = hotel1_countryhouse[143:153]
        hotel1_countryhouse_141to150 = hotel1_countryhouse[154:164]
        hotel1_countryhouse_151to160 = hotel1_countryhouse[165:175]
        hotel1_countryhouse_161to170 = hotel1_countryhouse[176:186]
        hotel1_countryhouse_171to180 = hotel1_countryhouse[187:197]
        hotel1_countryhouse_181to190 = hotel1_countryhouse[198:208]
        hotel1_countryhouse_191to200 = hotel1_countryhouse[209:219]
        hotel1_countryhouse_201to210 = hotel1_countryhouse[220:230]
        hotel1_countryhouse_211to220 = hotel1_countryhouse[231:241]

        hotel1_inn = Hotel.objects.filter(place=result[0], classfication='여관')
        hotel1_inn_1to10 = hotel1_inn[0:10]
        hotel1_inn_11to20 = hotel1_inn[11:21]
        hotel1_inn_21to30 = hotel1_inn[22:32]
        hotel1_inn_31to40 = hotel1_inn[33:43]
        hotel1_inn_41to50 = hotel1_inn[44:54]
        hotel1_inn_51to60 = hotel1_inn[55:65]
        hotel1_inn_61to70 = hotel1_inn[66:76]
        hotel1_inn_71to80 = hotel1_inn[77:87]
        hotel1_inn_81to90 = hotel1_inn[88:98]
        hotel1_inn_91to100 = hotel1_inn[99:109]
        hotel1_inn_101to110 = hotel1_inn[110:120]
        hotel1_inn_111to120 = hotel1_inn[121:131]
        hotel1_inn_121to130 = hotel1_inn[132:142]
        hotel1_inn_131to140 = hotel1_inn[143:153]
        hotel1_inn_141to150 = hotel1_inn[154:164]
        hotel1_inn_151to160 = hotel1_inn[165:175]
        hotel1_inn_161to170 = hotel1_inn[176:186]
        hotel1_inn_171to180 = hotel1_inn[187:197]
        hotel1_inn_181to190 = hotel1_inn[198:208]
        hotel1_inn_191to200 = hotel1_inn[209:219]
        hotel1_inn_201to210 = hotel1_inn[220:230]
        hotel1_inn_211to220 = hotel1_inn[231:241]

        hotel1_villa = Hotel.objects.filter(place=result[0],
                                            classfication='빌라')
        hotel1_villa_1to10 = hotel1_villa[0:10]
        hotel1_villa_11to20 = hotel1_villa[11:21]
        hotel1_villa_21to30 = hotel1_villa[22:32]
        hotel1_villa_31to40 = hotel1_villa[33:43]
        hotel1_villa_41to50 = hotel1_villa[44:54]
        hotel1_villa_51to60 = hotel1_villa[55:65]
        hotel1_villa_61to70 = hotel1_villa[66:76]
        hotel1_villa_71to80 = hotel1_villa[77:87]
        hotel1_villa_81to90 = hotel1_villa[88:98]
        hotel1_villa_91to100 = hotel1_villa[99:109]
        hotel1_villa_101to110 = hotel1_villa[110:120]
        hotel1_villa_111to120 = hotel1_villa[121:131]
        hotel1_villa_121to130 = hotel1_villa[132:142]
        hotel1_villa_131to140 = hotel1_villa[143:153]
        hotel1_villa_141to150 = hotel1_villa[154:164]
        hotel1_villa_151to160 = hotel1_villa[165:175]
        hotel1_villa_161to170 = hotel1_villa[176:186]
        hotel1_villa_171to180 = hotel1_villa[187:197]
        hotel1_villa_181to190 = hotel1_villa[198:208]
        hotel1_villa_191to200 = hotel1_villa[209:219]
        hotel1_villa_201to210 = hotel1_villa[220:230]
        hotel1_villa_211to220 = hotel1_villa[231:241]

        hotel1_camping = Hotel.objects.filter(place=result[0],
                                              classfication='캠핑장')
        hotel1_camping_1to10 = hotel1_camping[0:10]
        hotel1_camping_11to20 = hotel1_camping[11:21]
        hotel1_camping_21to30 = hotel1_camping[22:32]
        hotel1_camping_31to40 = hotel1_camping[33:43]
        hotel1_camping_41to50 = hotel1_camping[44:54]
        hotel1_camping_51to60 = hotel1_camping[55:65]
        hotel1_camping_61to70 = hotel1_camping[66:76]
        hotel1_camping_71to80 = hotel1_camping[77:87]
        hotel1_camping_81to90 = hotel1_camping[88:98]
        hotel1_camping_91to100 = hotel1_camping[99:109]
        hotel1_camping_101to110 = hotel1_camping[110:120]
        hotel1_camping_111to120 = hotel1_camping[121:131]
        hotel1_camping_121to130 = hotel1_camping[132:142]
        hotel1_camping_131to140 = hotel1_camping[143:153]
        hotel1_camping_141to150 = hotel1_camping[154:164]
        hotel1_camping_151to160 = hotel1_camping[165:175]
        hotel1_camping_161to170 = hotel1_camping[176:186]
        hotel1_camping_171to180 = hotel1_camping[187:197]
        hotel1_camping_181to190 = hotel1_camping[198:208]
        hotel1_camping_191to200 = hotel1_camping[209:219]
        hotel1_camping_201to210 = hotel1_camping[220:230]
        hotel1_camping_211to220 = hotel1_camping[231:241]

        hotel2 = Hotel.objects.filter(place=result[1])
        hotel2_1to10 = hotel2[0:10]
        hotel2_11to20 = hotel2[11:21]
        hotel2_21to30 = hotel2[22:32]
        hotel2_31to40 = hotel2[33:43]
        hotel2_41to50 = hotel2[44:54]
        hotel2_51to60 = hotel2[55:65]
        hotel2_61to70 = hotel2[66:76]
        hotel2_71to80 = hotel2[77:87]
        hotel2_81to90 = hotel2[88:98]
        hotel2_91to100 = hotel2[99:109]
        hotel2_101to110 = hotel2[110:120]
        hotel2_111to120 = hotel2[121:131]
        hotel2_121to130 = hotel2[132:142]
        hotel2_131to140 = hotel2[143:153]
        hotel2_141to150 = hotel2[154:164]
        hotel2_151to160 = hotel2[165:175]
        hotel2_161to170 = hotel2[176:186]
        hotel2_171to180 = hotel2[187:197]
        hotel2_181to190 = hotel2[198:208]
        hotel2_191to200 = hotel2[209:219]
        hotel2_201to210 = hotel2[220:230]
        hotel2_211to220 = hotel2[231:241]

        hotel2_cost_up = hotel2.order_by('cost')
        hotel2_cost_up_1to10 = hotel2_cost_up[0:10]
        hotel2_cost_up_11to20 = hotel2_cost_up[11:21]
        hotel2_cost_up_21to30 = hotel2_cost_up[22:32]
        hotel2_cost_up_31to40 = hotel2_cost_up[33:43]
        hotel2_cost_up_41to50 = hotel2_cost_up[44:54]
        hotel2_cost_up_51to60 = hotel2_cost_up[55:65]
        hotel2_cost_up_61to70 = hotel2_cost_up[66:76]
        hotel2_cost_up_71to80 = hotel2_cost_up[77:87]
        hotel2_cost_up_81to90 = hotel2_cost_up[88:98]
        hotel2_cost_up_91to100 = hotel2_cost_up[99:109]
        hotel2_cost_up_101to110 = hotel2_cost_up[110:120]
        hotel2_cost_up_111to120 = hotel2_cost_up[121:131]
        hotel2_cost_up_121to130 = hotel2_cost_up[132:142]
        hotel2_cost_up_131to140 = hotel2_cost_up[143:153]
        hotel2_cost_up_141to150 = hotel2_cost_up[154:164]
        hotel2_cost_up_151to160 = hotel2_cost_up[165:175]
        hotel2_cost_up_161to170 = hotel2_cost_up[176:186]
        hotel2_cost_up_171to180 = hotel2_cost_up[187:197]
        hotel2_cost_up_181to190 = hotel2_cost_up[198:208]
        hotel2_cost_up_191to200 = hotel2_cost_up[209:219]
        hotel2_cost_up_201to210 = hotel2_cost_up[220:230]
        hotel2_cost_up_211to220 = hotel2_cost_up[231:241]

        hotel2_cost_down = hotel2.order_by('-cost')
        hotel2_cost_down_1to10 = hotel2_cost_down[0:10]
        hotel2_cost_down_11to20 = hotel2_cost_down[11:21]
        hotel2_cost_down_21to30 = hotel2_cost_down[22:32]
        hotel2_cost_down_31to40 = hotel2_cost_down[33:43]
        hotel2_cost_down_41to50 = hotel2_cost_down[44:54]
        hotel2_cost_down_51to60 = hotel2_cost_down[55:65]
        hotel2_cost_down_61to70 = hotel2_cost_down[66:76]
        hotel2_cost_down_71to80 = hotel2_cost_down[77:87]
        hotel2_cost_down_81to90 = hotel2_cost_down[88:98]
        hotel2_cost_down_91to100 = hotel2_cost_down[99:109]
        hotel2_cost_down_101to110 = hotel2_cost_down[110:120]
        hotel2_cost_down_111to120 = hotel2_cost_down[121:131]
        hotel2_cost_down_121to130 = hotel2_cost_down[132:142]
        hotel2_cost_down_131to140 = hotel2_cost_down[143:153]
        hotel2_cost_down_141to150 = hotel2_cost_down[154:164]
        hotel2_cost_down_151to160 = hotel2_cost_down[165:175]
        hotel2_cost_down_161to170 = hotel2_cost_down[176:186]
        hotel2_cost_down_171to180 = hotel2_cost_down[187:197]
        hotel2_cost_down_181to190 = hotel2_cost_down[198:208]
        hotel2_cost_down_191to200 = hotel2_cost_down[209:219]
        hotel2_cost_down_201to210 = hotel2_cost_down[220:230]
        hotel2_cost_down_211to220 = hotel2_cost_down[231:241]

        hotel2_rating_down = hotel2.order_by('-rating')
        hotel2_rating_down_1to10 = hotel2_rating_down[0:10]
        hotel2_rating_down_11to20 = hotel2_rating_down[11:21]
        hotel2_rating_down_21to30 = hotel2_rating_down[22:32]
        hotel2_rating_down_31to40 = hotel2_rating_down[33:43]
        hotel2_rating_down_41to50 = hotel2_rating_down[44:54]
        hotel2_rating_down_51to60 = hotel2_rating_down[55:65]
        hotel2_rating_down_61to70 = hotel2_rating_down[66:76]
        hotel2_rating_down_71to80 = hotel2_rating_down[77:87]
        hotel2_rating_down_81to90 = hotel2_rating_down[88:98]
        hotel2_rating_down_91to100 = hotel2_rating_down[99:109]
        hotel2_rating_down_101to110 = hotel2_rating_down[110:120]
        hotel2_rating_down_111to120 = hotel2_rating_down[121:131]
        hotel2_rating_down_121to130 = hotel2_rating_down[132:142]
        hotel2_rating_down_131to140 = hotel2_rating_down[143:153]
        hotel2_rating_down_141to150 = hotel2_rating_down[154:164]
        hotel2_rating_down_151to160 = hotel2_rating_down[165:175]
        hotel2_rating_down_161to170 = hotel2_rating_down[176:186]
        hotel2_rating_down_171to180 = hotel2_rating_down[187:197]
        hotel2_rating_down_181to190 = hotel2_rating_down[198:208]
        hotel2_rating_down_191to200 = hotel2_rating_down[209:219]
        hotel2_rating_down_201to210 = hotel2_rating_down[220:230]
        hotel2_rating_down_211to220 = hotel2_rating_down[231:241]

        hotel2_distance_up = hotel2.order_by('distance')
        hotel2_distance_up_1to10 = hotel2_distance_up[0:10]
        hotel2_distance_up_11to20 = hotel2_distance_up[11:21]
        hotel2_distance_up_21to30 = hotel2_distance_up[22:32]
        hotel2_distance_up_31to40 = hotel2_distance_up[33:43]
        hotel2_distance_up_41to50 = hotel2_distance_up[44:54]
        hotel2_distance_up_51to60 = hotel2_distance_up[55:65]
        hotel2_distance_up_61to70 = hotel2_distance_up[66:76]
        hotel2_distance_up_71to80 = hotel2_distance_up[77:87]
        hotel2_distance_up_81to90 = hotel2_distance_up[88:98]
        hotel2_distance_up_91to100 = hotel2_distance_up[99:109]
        hotel2_distance_up_101to110 = hotel2_distance_up[110:120]
        hotel2_distance_up_111to120 = hotel2_distance_up[121:131]
        hotel2_distance_up_121to130 = hotel2_distance_up[132:142]
        hotel2_distance_up_131to140 = hotel2_distance_up[143:153]
        hotel2_distance_up_141to150 = hotel2_distance_up[154:164]
        hotel2_distance_up_151to160 = hotel2_distance_up[165:175]
        hotel2_distance_up_161to170 = hotel2_distance_up[176:186]
        hotel2_distance_up_171to180 = hotel2_distance_up[187:197]
        hotel2_distance_up_181to190 = hotel2_distance_up[198:208]
        hotel2_distance_up_191to200 = hotel2_distance_up[209:219]
        hotel2_distance_up_201to210 = hotel2_distance_up[220:230]
        hotel2_distance_up_211to220 = hotel2_distance_up[231:241]

        hotel2_kind_down = hotel2.order_by('-kind')
        hotel2_kind_down_1to10 = hotel2_kind_down[0:10]
        hotel2_kind_down_11to20 = hotel2_kind_down[11:21]
        hotel2_kind_down_21to30 = hotel2_kind_down[22:32]
        hotel2_kind_down_31to40 = hotel2_kind_down[33:43]
        hotel2_kind_down_41to50 = hotel2_kind_down[44:54]
        hotel2_kind_down_51to60 = hotel2_kind_down[55:65]
        hotel2_kind_down_61to70 = hotel2_kind_down[66:76]
        hotel2_kind_down_71to80 = hotel2_kind_down[77:87]
        hotel2_kind_down_81to90 = hotel2_kind_down[88:98]
        hotel2_kind_down_91to100 = hotel2_kind_down[99:109]
        hotel2_kind_down_101to110 = hotel2_kind_down[110:120]
        hotel2_kind_down_111to120 = hotel2_kind_down[121:131]
        hotel2_kind_down_121to130 = hotel2_kind_down[132:142]
        hotel2_kind_down_131to140 = hotel2_kind_down[143:153]
        hotel2_kind_down_141to150 = hotel2_kind_down[154:164]
        hotel2_kind_down_151to160 = hotel2_kind_down[165:175]
        hotel2_kind_down_161to170 = hotel2_kind_down[176:186]
        hotel2_kind_down_171to180 = hotel2_kind_down[187:197]
        hotel2_kind_down_181to190 = hotel2_kind_down[198:208]
        hotel2_kind_down_191to200 = hotel2_kind_down[209:219]
        hotel2_kind_down_201to210 = hotel2_kind_down[220:230]
        hotel2_kind_down_211to220 = hotel2_kind_down[231:241]

        hotel2_clean_down = hotel2.order_by('-clean')
        hotel2_clean_down_1to10 = hotel2_clean_down[0:10]
        hotel2_clean_down_11to20 = hotel2_clean_down[11:21]
        hotel2_clean_down_21to30 = hotel2_clean_down[22:32]
        hotel2_clean_down_31to40 = hotel2_clean_down[33:43]
        hotel2_clean_down_41to50 = hotel2_clean_down[44:54]
        hotel2_clean_down_51to60 = hotel2_clean_down[55:65]
        hotel2_clean_down_61to70 = hotel2_clean_down[66:76]
        hotel2_clean_down_71to80 = hotel2_clean_down[77:87]
        hotel2_clean_down_81to90 = hotel2_clean_down[88:98]
        hotel2_clean_down_91to100 = hotel2_clean_down[99:109]
        hotel2_clean_down_101to110 = hotel2_clean_down[110:120]
        hotel2_clean_down_111to120 = hotel2_clean_down[121:131]
        hotel2_clean_down_121to130 = hotel2_clean_down[132:142]
        hotel2_clean_down_131to140 = hotel2_clean_down[143:153]
        hotel2_clean_down_141to150 = hotel2_clean_down[154:164]
        hotel2_clean_down_151to160 = hotel2_clean_down[165:175]
        hotel2_clean_down_161to170 = hotel2_clean_down[176:186]
        hotel2_clean_down_171to180 = hotel2_clean_down[187:197]
        hotel2_clean_down_181to190 = hotel2_clean_down[198:208]
        hotel2_clean_down_191to200 = hotel2_clean_down[209:219]
        hotel2_clean_down_201to210 = hotel2_clean_down[220:230]
        hotel2_clean_down_211to220 = hotel2_clean_down[231:241]

        hotel2_conv_down = hotel2.order_by('-conv')
        hotel2_conv_down_1to10 = hotel2_conv_down[0:10]
        hotel2_conv_down_11to20 = hotel2_conv_down[11:21]
        hotel2_conv_down_21to30 = hotel2_conv_down[22:32]
        hotel2_conv_down_31to40 = hotel2_conv_down[33:43]
        hotel2_conv_down_41to50 = hotel2_conv_down[44:54]
        hotel2_conv_down_51to60 = hotel2_conv_down[55:65]
        hotel2_conv_down_61to70 = hotel2_conv_down[66:76]
        hotel2_conv_down_71to80 = hotel2_conv_down[77:87]
        hotel2_conv_down_81to90 = hotel2_conv_down[88:98]
        hotel2_conv_down_91to100 = hotel2_conv_down[99:109]
        hotel2_conv_down_101to110 = hotel2_conv_down[110:120]
        hotel2_conv_down_111to120 = hotel2_conv_down[121:131]
        hotel2_conv_down_121to130 = hotel2_conv_down[132:142]
        hotel2_conv_down_131to140 = hotel2_conv_down[143:153]
        hotel2_conv_down_141to150 = hotel2_conv_down[154:164]
        hotel2_conv_down_151to160 = hotel2_conv_down[165:175]
        hotel2_conv_down_161to170 = hotel2_conv_down[176:186]
        hotel2_conv_down_171to180 = hotel2_conv_down[187:197]
        hotel2_conv_down_181to190 = hotel2_conv_down[198:208]
        hotel2_conv_down_191to200 = hotel2_conv_down[209:219]
        hotel2_conv_down_201to210 = hotel2_conv_down[220:230]
        hotel2_conv_down_211to220 = hotel2_conv_down[231:241]

        hotel2_hotel = Hotel.objects.filter(place=result[1],
                                            classfication='호텔')
        hotel2_hotel_1to10 = hotel2_hotel[0:10]
        hotel2_hotel_11to20 = hotel2_hotel[11:21]
        hotel2_hotel_21to30 = hotel2_hotel[22:32]
        hotel2_hotel_31to40 = hotel2_hotel[33:43]
        hotel2_hotel_41to50 = hotel2_hotel[44:54]
        hotel2_hotel_51to60 = hotel2_hotel[55:65]
        hotel2_hotel_61to70 = hotel2_hotel[66:76]
        hotel2_hotel_71to80 = hotel2_hotel[77:87]
        hotel2_hotel_81to90 = hotel2_hotel[88:98]
        hotel2_hotel_91to100 = hotel2_hotel[99:109]
        hotel2_hotel_101to110 = hotel2_hotel[110:120]
        hotel2_hotel_111to120 = hotel2_hotel[121:131]
        hotel2_hotel_121to130 = hotel2_hotel[132:142]
        hotel2_hotel_131to140 = hotel2_hotel[143:153]
        hotel2_hotel_141to150 = hotel2_hotel[154:164]
        hotel2_hotel_151to160 = hotel2_hotel[165:175]
        hotel2_hotel_161to170 = hotel2_hotel[176:186]
        hotel2_hotel_171to180 = hotel2_hotel[187:197]
        hotel2_hotel_181to190 = hotel2_hotel[198:208]
        hotel2_hotel_191to200 = hotel2_hotel[209:219]
        hotel2_hotel_201to210 = hotel2_hotel[220:230]
        hotel2_hotel_211to220 = hotel2_hotel[231:241]

        hotel2_hostel = Hotel.objects.filter(place=result[1],
                                             classfication='호스텔')
        hotel2_hostel_1to10 = hotel2_hostel[0:10]
        hotel2_hostel_11to20 = hotel2_hostel[11:21]
        hotel2_hostel_21to30 = hotel2_hostel[22:32]
        hotel2_hostel_31to40 = hotel2_hostel[33:43]
        hotel2_hostel_41to50 = hotel2_hostel[44:54]
        hotel2_hostel_51to60 = hotel2_hostel[55:65]
        hotel2_hostel_61to70 = hotel2_hostel[66:76]
        hotel2_hostel_71to80 = hotel2_hostel[77:87]
        hotel2_hostel_81to90 = hotel2_hostel[88:98]
        hotel2_hostel_91to100 = hotel2_hostel[99:109]
        hotel2_hostel_101to110 = hotel2_hostel[110:120]
        hotel2_hostel_111to120 = hotel2_hostel[121:131]
        hotel2_hostel_121to130 = hotel2_hostel[132:142]
        hotel2_hostel_131to140 = hotel2_hostel[143:153]
        hotel2_hostel_141to150 = hotel2_hostel[154:164]
        hotel2_hostel_151to160 = hotel2_hostel[165:175]
        hotel2_hostel_161to170 = hotel2_hostel[176:186]
        hotel2_hostel_171to180 = hotel2_hostel[187:197]
        hotel2_hostel_181to190 = hotel2_hostel[198:208]
        hotel2_hostel_191to200 = hotel2_hostel[209:219]
        hotel2_hostel_201to210 = hotel2_hostel[220:230]
        hotel2_hostel_211to220 = hotel2_hostel[231:241]

        hotel2_guest = Hotel.objects.filter(place=result[1],
                                            classfication='게스트하우스')
        hotel2_guest_1to10 = hotel2_guest[0:10]
        hotel2_guest_11to20 = hotel2_guest[11:21]
        hotel2_guest_21to30 = hotel2_guest[22:32]
        hotel2_guest_31to40 = hotel2_guest[33:43]
        hotel2_guest_41to50 = hotel2_guest[44:54]
        hotel2_guest_51to60 = hotel2_guest[55:65]
        hotel2_guest_61to70 = hotel2_guest[66:76]
        hotel2_guest_71to80 = hotel2_guest[77:87]
        hotel2_guest_81to90 = hotel2_guest[88:98]
        hotel2_guest_91to100 = hotel2_guest[99:109]
        hotel2_guest_101to110 = hotel2_guest[110:120]
        hotel2_guest_111to120 = hotel2_guest[121:131]
        hotel2_guest_121to130 = hotel2_guest[132:142]
        hotel2_guest_131to140 = hotel2_guest[143:153]
        hotel2_guest_141to150 = hotel2_guest[154:164]
        hotel2_guest_151to160 = hotel2_guest[165:175]
        hotel2_guest_161to170 = hotel2_guest[176:186]
        hotel2_guest_171to180 = hotel2_guest[187:197]
        hotel2_guest_181to190 = hotel2_guest[198:208]
        hotel2_guest_191to200 = hotel2_guest[209:219]
        hotel2_guest_201to210 = hotel2_guest[220:230]
        hotel2_guest_211to220 = hotel2_guest[231:241]

        hotel2_apartment = Hotel.objects.filter(place=result[1],
                                                classfication='아파트')
        hotel2_apartment_1to10 = hotel2_apartment[0:10]
        hotel2_apartment_11to20 = hotel2_apartment[11:21]
        hotel2_apartment_21to30 = hotel2_apartment[22:32]
        hotel2_apartment_31to40 = hotel2_apartment[33:43]
        hotel2_apartment_41to50 = hotel2_apartment[44:54]
        hotel2_apartment_51to60 = hotel2_apartment[55:65]
        hotel2_apartment_61to70 = hotel2_apartment[66:76]
        hotel2_apartment_71to80 = hotel2_apartment[77:87]
        hotel2_apartment_81to90 = hotel2_apartment[88:98]
        hotel2_apartment_91to100 = hotel2_apartment[99:109]
        hotel2_apartment_101to110 = hotel2_apartment[110:120]
        hotel2_apartment_111to120 = hotel2_apartment[121:131]
        hotel2_apartment_121to130 = hotel2_apartment[132:142]
        hotel2_apartment_131to140 = hotel2_apartment[143:153]
        hotel2_apartment_141to150 = hotel2_apartment[154:164]
        hotel2_apartment_151to160 = hotel2_apartment[165:175]
        hotel2_apartment_161to170 = hotel2_apartment[176:186]
        hotel2_apartment_171to180 = hotel2_apartment[187:197]
        hotel2_apartment_181to190 = hotel2_apartment[198:208]
        hotel2_apartment_191to200 = hotel2_apartment[209:219]
        hotel2_apartment_201to210 = hotel2_apartment[220:230]
        hotel2_apartment_211to220 = hotel2_apartment[231:241]

        hotel2_apartmenthotel = Hotel.objects.filter(place=result[1],
                                                     classfication='아파트호텔')
        hotel2_apartmenthotel_1to10 = hotel2_apartmenthotel[0:10]
        hotel2_apartmenthotel_11to20 = hotel2_apartmenthotel[11:21]
        hotel2_apartmenthotel_21to30 = hotel2_apartmenthotel[22:32]
        hotel2_apartmenthotel_31to40 = hotel2_apartmenthotel[33:43]
        hotel2_apartmenthotel_41to50 = hotel2_apartmenthotel[44:54]
        hotel2_apartmenthotel_51to60 = hotel2_apartmenthotel[55:65]
        hotel2_apartmenthotel_61to70 = hotel2_apartmenthotel[66:76]
        hotel2_apartmenthotel_71to80 = hotel2_apartmenthotel[77:87]
        hotel2_apartmenthotel_81to90 = hotel2_apartmenthotel[88:98]
        hotel2_apartmenthotel_91to100 = hotel2_apartmenthotel[99:109]
        hotel2_apartmenthotel_101to110 = hotel2_apartmenthotel[110:120]
        hotel2_apartmenthotel_111to120 = hotel2_apartmenthotel[121:131]
        hotel2_apartmenthotel_121to130 = hotel2_apartmenthotel[132:142]
        hotel2_apartmenthotel_131to140 = hotel2_apartmenthotel[143:153]
        hotel2_apartmenthotel_141to150 = hotel2_apartmenthotel[154:164]
        hotel2_apartmenthotel_151to160 = hotel2_apartmenthotel[165:175]
        hotel2_apartmenthotel_161to170 = hotel2_apartmenthotel[176:186]
        hotel2_apartmenthotel_171to180 = hotel2_apartmenthotel[187:197]
        hotel2_apartmenthotel_181to190 = hotel2_apartmenthotel[198:208]
        hotel2_apartmenthotel_191to200 = hotel2_apartmenthotel[209:219]
        hotel2_apartmenthotel_201to210 = hotel2_apartmenthotel[220:230]
        hotel2_apartmenthotel_211to220 = hotel2_apartmenthotel[231:241]

        hotel2_motel = Hotel.objects.filter(place=result[1],
                                            classfication='모텔')
        hotel2_motel_1to10 = hotel2_motel[0:10]
        hotel2_motel_11to20 = hotel2_motel[11:21]
        hotel2_motel_21to30 = hotel2_motel[22:32]
        hotel2_motel_31to40 = hotel2_motel[33:43]
        hotel2_motel_41to50 = hotel2_motel[44:54]
        hotel2_motel_51to60 = hotel2_motel[55:65]
        hotel2_motel_61to70 = hotel2_motel[66:76]
        hotel2_motel_71to80 = hotel2_motel[77:87]
        hotel2_motel_81to90 = hotel2_motel[88:98]
        hotel2_motel_91to100 = hotel2_motel[99:109]
        hotel2_motel_101to110 = hotel2_motel[110:120]
        hotel2_motel_111to120 = hotel2_motel[121:131]
        hotel2_motel_121to130 = hotel2_motel[132:142]
        hotel2_motel_131to140 = hotel2_motel[143:153]
        hotel2_motel_141to150 = hotel2_motel[154:164]
        hotel2_motel_151to160 = hotel2_motel[165:175]
        hotel2_motel_161to170 = hotel2_motel[176:186]
        hotel2_motel_171to180 = hotel2_motel[187:197]
        hotel2_motel_181to190 = hotel2_motel[198:208]
        hotel2_motel_191to200 = hotel2_motel[209:219]
        hotel2_motel_201to210 = hotel2_motel[220:230]
        hotel2_motel_211to220 = hotel2_motel[231:241]

        hotel2_pension = Hotel.objects.filter(place=result[1],
                                              classfication='펜션')
        hotel2_pension_1to10 = hotel2_pension[0:10]
        hotel2_pension_11to20 = hotel2_pension[11:21]
        hotel2_pension_21to30 = hotel2_pension[22:32]
        hotel2_pension_31to40 = hotel2_pension[33:43]
        hotel2_pension_41to50 = hotel2_pension[44:54]
        hotel2_pension_51to60 = hotel2_pension[55:65]
        hotel2_pension_61to70 = hotel2_pension[66:76]
        hotel2_pension_71to80 = hotel2_pension[77:87]
        hotel2_pension_81to90 = hotel2_pension[88:98]
        hotel2_pension_91to100 = hotel2_pension[99:109]
        hotel2_pension_101to110 = hotel2_pension[110:120]
        hotel2_pension_111to120 = hotel2_pension[121:131]
        hotel2_pension_121to130 = hotel2_pension[132:142]
        hotel2_pension_131to140 = hotel2_pension[143:153]
        hotel2_pension_141to150 = hotel2_pension[154:164]
        hotel2_pension_151to160 = hotel2_pension[165:175]
        hotel2_pension_161to170 = hotel2_pension[176:186]
        hotel2_pension_171to180 = hotel2_pension[187:197]
        hotel2_pension_181to190 = hotel2_pension[198:208]
        hotel2_pension_191to200 = hotel2_pension[209:219]
        hotel2_pension_201to210 = hotel2_pension[220:230]
        hotel2_pension_211to220 = hotel2_pension[231:241]

        hotel2_resort = Hotel.objects.filter(place=result[1],
                                             classfication='리조트')
        hotel2_resort_1to10 = hotel2_resort[0:10]
        hotel2_resort_11to20 = hotel2_resort[11:21]
        hotel2_resort_21to30 = hotel2_resort[22:32]
        hotel2_resort_31to40 = hotel2_resort[33:43]
        hotel2_resort_41to50 = hotel2_resort[44:54]
        hotel2_resort_51to60 = hotel2_resort[55:65]
        hotel2_resort_61to70 = hotel2_resort[66:76]
        hotel2_resort_71to80 = hotel2_resort[77:87]
        hotel2_resort_81to90 = hotel2_resort[88:98]
        hotel2_resort_91to100 = hotel2_resort[99:109]
        hotel2_resort_101to110 = hotel2_resort[110:120]
        hotel2_resort_111to120 = hotel2_resort[121:131]
        hotel2_resort_121to130 = hotel2_resort[132:142]
        hotel2_resort_131to140 = hotel2_resort[143:153]
        hotel2_resort_141to150 = hotel2_resort[154:164]
        hotel2_resort_151to160 = hotel2_resort[165:175]
        hotel2_resort_161to170 = hotel2_resort[176:186]
        hotel2_resort_171to180 = hotel2_resort[187:197]
        hotel2_resort_181to190 = hotel2_resort[198:208]
        hotel2_resort_191to200 = hotel2_resort[209:219]
        hotel2_resort_201to210 = hotel2_resort[220:230]
        hotel2_resort_211to220 = hotel2_resort[231:241]

        hotel2_badandbreakfast = Hotel.objects.filter(place=result[1],
                                                      classfication='베드앤브렉퍼스트')
        hotel2_badandbreakfast_1to10 = hotel2_badandbreakfast[0:10]
        hotel2_badandbreakfast_11to20 = hotel2_badandbreakfast[11:21]
        hotel2_badandbreakfast_21to30 = hotel2_badandbreakfast[22:32]
        hotel2_badandbreakfast_31to40 = hotel2_badandbreakfast[33:43]
        hotel2_badandbreakfast_41to50 = hotel2_badandbreakfast[44:54]
        hotel2_badandbreakfast_51to60 = hotel2_badandbreakfast[55:65]
        hotel2_badandbreakfast_61to70 = hotel2_badandbreakfast[66:76]
        hotel2_badandbreakfast_71to80 = hotel2_badandbreakfast[77:87]
        hotel2_badandbreakfast_81to90 = hotel2_badandbreakfast[88:98]
        hotel2_badandbreakfast_91to100 = hotel2_badandbreakfast[99:109]
        hotel2_badandbreakfast_101to110 = hotel2_badandbreakfast[110:120]
        hotel2_badandbreakfast_111to120 = hotel2_badandbreakfast[121:131]
        hotel2_badandbreakfast_121to130 = hotel2_badandbreakfast[132:142]
        hotel2_badandbreakfast_131to140 = hotel2_badandbreakfast[143:153]
        hotel2_badandbreakfast_141to150 = hotel2_badandbreakfast[154:164]
        hotel2_badandbreakfast_151to160 = hotel2_badandbreakfast[165:175]
        hotel2_badandbreakfast_161to170 = hotel2_badandbreakfast[176:186]
        hotel2_badandbreakfast_171to180 = hotel2_badandbreakfast[187:197]
        hotel2_badandbreakfast_181to190 = hotel2_badandbreakfast[198:208]
        hotel2_badandbreakfast_191to200 = hotel2_badandbreakfast[209:219]
        hotel2_badandbreakfast_201to210 = hotel2_badandbreakfast[220:230]
        hotel2_badandbreakfast_211to220 = hotel2_badandbreakfast[231:241]

        hotel2_homestay = Hotel.objects.filter(place=result[1],
                                               classfication='홈스테이')
        hotel2_homestay_1to10 = hotel2_homestay[0:10]
        hotel2_homestay_11to20 = hotel2_homestay[11:21]
        hotel2_homestay_21to30 = hotel2_homestay[22:32]
        hotel2_homestay_31to40 = hotel2_homestay[33:43]
        hotel2_homestay_41to50 = hotel2_homestay[44:54]
        hotel2_homestay_51to60 = hotel2_homestay[55:65]
        hotel2_homestay_61to70 = hotel2_homestay[66:76]
        hotel2_homestay_71to80 = hotel2_homestay[77:87]
        hotel2_homestay_81to90 = hotel2_homestay[88:98]
        hotel2_homestay_91to100 = hotel2_homestay[99:109]
        hotel2_homestay_101to110 = hotel2_homestay[110:120]
        hotel2_homestay_111to120 = hotel2_homestay[121:131]
        hotel2_homestay_121to130 = hotel2_homestay[132:142]
        hotel2_homestay_131to140 = hotel2_homestay[143:153]
        hotel2_homestay_141to150 = hotel2_homestay[154:164]
        hotel2_homestay_151to160 = hotel2_homestay[165:175]
        hotel2_homestay_161to170 = hotel2_homestay[176:186]
        hotel2_homestay_171to180 = hotel2_homestay[187:197]
        hotel2_homestay_181to190 = hotel2_homestay[198:208]
        hotel2_homestay_191to200 = hotel2_homestay[209:219]
        hotel2_homestay_201to210 = hotel2_homestay[220:230]
        hotel2_homestay_211to220 = hotel2_homestay[231:241]

        hotel2_lodge = Hotel.objects.filter(place=result[1],
                                            classfication='롯지')
        hotel2_lodge_1to10 = hotel2_lodge[0:10]
        hotel2_lodge_11to20 = hotel2_lodge[11:21]
        hotel2_lodge_21to30 = hotel2_lodge[22:32]
        hotel2_lodge_31to40 = hotel2_lodge[33:43]
        hotel2_lodge_41to50 = hotel2_lodge[44:54]
        hotel2_lodge_51to60 = hotel2_lodge[55:65]
        hotel2_lodge_61to70 = hotel2_lodge[66:76]
        hotel2_lodge_71to80 = hotel2_lodge[77:87]
        hotel2_lodge_81to90 = hotel2_lodge[88:98]
        hotel2_lodge_91to100 = hotel2_lodge[99:109]
        hotel2_lodge_101to110 = hotel2_lodge[110:120]
        hotel2_lodge_111to120 = hotel2_lodge[121:131]
        hotel2_lodge_121to130 = hotel2_lodge[132:142]
        hotel2_lodge_131to140 = hotel2_lodge[143:153]
        hotel2_lodge_141to150 = hotel2_lodge[154:164]
        hotel2_lodge_151to160 = hotel2_lodge[165:175]
        hotel2_lodge_161to170 = hotel2_lodge[176:186]
        hotel2_lodge_171to180 = hotel2_lodge[187:197]
        hotel2_lodge_181to190 = hotel2_lodge[198:208]
        hotel2_lodge_191to200 = hotel2_lodge[209:219]
        hotel2_lodge_201to210 = hotel2_lodge[220:230]
        hotel2_lodge_211to220 = hotel2_lodge[231:241]

        hotel2_countryhouse = Hotel.objects.filter(place=result[1],
                                                   classfication='컨트리하우스')
        hotel2_countryhouse_1to10 = hotel2_countryhouse[0:10]
        hotel2_countryhouse_11to20 = hotel2_countryhouse[11:21]
        hotel2_countryhouse_21to30 = hotel2_countryhouse[22:32]
        hotel2_countryhouse_31to40 = hotel2_countryhouse[33:43]
        hotel2_countryhouse_41to50 = hotel2_countryhouse[44:54]
        hotel2_countryhouse_51to60 = hotel2_countryhouse[55:65]
        hotel2_countryhouse_61to70 = hotel2_countryhouse[66:76]
        hotel2_countryhouse_71to80 = hotel2_countryhouse[77:87]
        hotel2_countryhouse_81to90 = hotel2_countryhouse[88:98]
        hotel2_countryhouse_91to100 = hotel2_countryhouse[99:109]
        hotel2_countryhouse_101to110 = hotel2_countryhouse[110:120]
        hotel2_countryhouse_111to120 = hotel2_countryhouse[121:131]
        hotel2_countryhouse_121to130 = hotel2_countryhouse[132:142]
        hotel2_countryhouse_131to140 = hotel2_countryhouse[143:153]
        hotel2_countryhouse_141to150 = hotel2_countryhouse[154:164]
        hotel2_countryhouse_151to160 = hotel2_countryhouse[165:175]
        hotel2_countryhouse_161to170 = hotel2_countryhouse[176:186]
        hotel2_countryhouse_171to180 = hotel2_countryhouse[187:197]
        hotel2_countryhouse_181to190 = hotel2_countryhouse[198:208]
        hotel2_countryhouse_191to200 = hotel2_countryhouse[209:219]
        hotel2_countryhouse_201to210 = hotel2_countryhouse[220:230]
        hotel2_countryhouse_211to220 = hotel2_countryhouse[231:241]

        hotel2_inn = Hotel.objects.filter(place=result[1], classfication='여관')
        hotel2_inn_1to10 = hotel2_inn[0:10]
        hotel2_inn_11to20 = hotel2_inn[11:21]
        hotel2_inn_21to30 = hotel2_inn[22:32]
        hotel2_inn_31to40 = hotel2_inn[33:43]
        hotel2_inn_41to50 = hotel2_inn[44:54]
        hotel2_inn_51to60 = hotel2_inn[55:65]
        hotel2_inn_61to70 = hotel2_inn[66:76]
        hotel2_inn_71to80 = hotel2_inn[77:87]
        hotel2_inn_81to90 = hotel2_inn[88:98]
        hotel2_inn_91to100 = hotel2_inn[99:109]
        hotel2_inn_101to110 = hotel2_inn[110:120]
        hotel2_inn_111to120 = hotel2_inn[121:131]
        hotel2_inn_121to130 = hotel2_inn[132:142]
        hotel2_inn_131to140 = hotel2_inn[143:153]
        hotel2_inn_141to150 = hotel2_inn[154:164]
        hotel2_inn_151to160 = hotel2_inn[165:175]
        hotel2_inn_161to170 = hotel2_inn[176:186]
        hotel2_inn_171to180 = hotel2_inn[187:197]
        hotel2_inn_181to190 = hotel2_inn[198:208]
        hotel2_inn_191to200 = hotel2_inn[209:219]
        hotel2_inn_201to210 = hotel2_inn[220:230]
        hotel2_inn_211to220 = hotel2_inn[231:241]

        hotel2_villa = Hotel.objects.filter(place=result[1],
                                            classfication='빌라')
        hotel2_villa_1to10 = hotel2_villa[0:10]
        hotel2_villa_11to20 = hotel2_villa[11:21]
        hotel2_villa_21to30 = hotel2_villa[22:32]
        hotel2_villa_31to40 = hotel2_villa[33:43]
        hotel2_villa_41to50 = hotel2_villa[44:54]
        hotel2_villa_51to60 = hotel2_villa[55:65]
        hotel2_villa_61to70 = hotel2_villa[66:76]
        hotel2_villa_71to80 = hotel2_villa[77:87]
        hotel2_villa_81to90 = hotel2_villa[88:98]
        hotel2_villa_91to100 = hotel2_villa[99:109]
        hotel2_villa_101to110 = hotel2_villa[110:120]
        hotel2_villa_111to120 = hotel2_villa[121:131]
        hotel2_villa_121to130 = hotel2_villa[132:142]
        hotel2_villa_131to140 = hotel2_villa[143:153]
        hotel2_villa_141to150 = hotel2_villa[154:164]
        hotel2_villa_151to160 = hotel2_villa[165:175]
        hotel2_villa_161to170 = hotel2_villa[176:186]
        hotel2_villa_171to180 = hotel2_villa[187:197]
        hotel2_villa_181to190 = hotel2_villa[198:208]
        hotel2_villa_191to200 = hotel2_villa[209:219]
        hotel2_villa_201to210 = hotel2_villa[220:230]
        hotel2_villa_211to220 = hotel2_villa[231:241]

        hotel2_camping = Hotel.objects.filter(place=result[1],
                                              classfication='캠핑장')
        hotel2_camping_1to10 = hotel2_camping[0:10]
        hotel2_camping_11to20 = hotel2_camping[11:21]
        hotel2_camping_21to30 = hotel2_camping[22:32]
        hotel2_camping_31to40 = hotel2_camping[33:43]
        hotel2_camping_41to50 = hotel2_camping[44:54]
        hotel2_camping_51to60 = hotel2_camping[55:65]
        hotel2_camping_61to70 = hotel2_camping[66:76]
        hotel2_camping_71to80 = hotel2_camping[77:87]
        hotel2_camping_81to90 = hotel2_camping[88:98]
        hotel2_camping_91to100 = hotel2_camping[99:109]
        hotel2_camping_101to110 = hotel2_camping[110:120]
        hotel2_camping_111to120 = hotel2_camping[121:131]
        hotel2_camping_121to130 = hotel2_camping[132:142]
        hotel2_camping_131to140 = hotel2_camping[143:153]
        hotel2_camping_141to150 = hotel2_camping[154:164]
        hotel2_camping_151to160 = hotel2_camping[165:175]
        hotel2_camping_161to170 = hotel2_camping[176:186]
        hotel2_camping_171to180 = hotel2_camping[187:197]
        hotel2_camping_181to190 = hotel2_camping[198:208]
        hotel2_camping_191to200 = hotel2_camping[209:219]
        hotel2_camping_201to210 = hotel2_camping[220:230]
        hotel2_camping_211to220 = hotel2_camping[231:241]

        hotel3 = Hotel.objects.filter(place=result[2])
        hotel3_1to10 = hotel3[0:10]
        hotel3_11to20 = hotel3[11:21]
        hotel3_21to30 = hotel3[22:32]
        hotel3_31to40 = hotel3[33:43]
        hotel3_41to50 = hotel3[44:54]
        hotel3_51to60 = hotel3[55:65]
        hotel3_61to70 = hotel3[66:76]
        hotel3_71to80 = hotel3[77:87]
        hotel3_81to90 = hotel3[88:98]
        hotel3_91to100 = hotel3[99:109]
        hotel3_101to110 = hotel3[110:120]
        hotel3_111to120 = hotel3[121:131]
        hotel3_121to130 = hotel3[132:142]
        hotel3_131to140 = hotel3[143:153]
        hotel3_141to150 = hotel3[154:164]
        hotel3_151to160 = hotel3[165:175]
        hotel3_161to170 = hotel3[176:186]
        hotel3_171to180 = hotel3[187:197]
        hotel3_181to190 = hotel3[198:208]
        hotel3_191to200 = hotel3[209:219]
        hotel3_201to210 = hotel3[220:230]
        hotel3_211to220 = hotel3[231:241]

        hotel3_cost_up = hotel3.order_by('cost')
        hotel3_cost_up_1to10 = hotel3_cost_up[0:10]
        hotel3_cost_up_11to20 = hotel3_cost_up[11:21]
        hotel3_cost_up_21to30 = hotel3_cost_up[22:32]
        hotel3_cost_up_31to40 = hotel3_cost_up[33:43]
        hotel3_cost_up_41to50 = hotel3_cost_up[44:54]
        hotel3_cost_up_51to60 = hotel3_cost_up[55:65]
        hotel3_cost_up_61to70 = hotel3_cost_up[66:76]
        hotel3_cost_up_71to80 = hotel3_cost_up[77:87]
        hotel3_cost_up_81to90 = hotel3_cost_up[88:98]
        hotel3_cost_up_91to100 = hotel3_cost_up[99:109]
        hotel3_cost_up_101to110 = hotel3_cost_up[110:120]
        hotel3_cost_up_111to120 = hotel3_cost_up[121:131]
        hotel3_cost_up_121to130 = hotel3_cost_up[132:142]
        hotel3_cost_up_131to140 = hotel3_cost_up[143:153]
        hotel3_cost_up_141to150 = hotel3_cost_up[154:164]
        hotel3_cost_up_151to160 = hotel3_cost_up[165:175]
        hotel3_cost_up_161to170 = hotel3_cost_up[176:186]
        hotel3_cost_up_171to180 = hotel3_cost_up[187:197]
        hotel3_cost_up_181to190 = hotel3_cost_up[198:208]
        hotel3_cost_up_191to200 = hotel3_cost_up[209:219]
        hotel3_cost_up_201to210 = hotel3_cost_up[220:230]
        hotel3_cost_up_211to220 = hotel3_cost_up[231:241]

        hotel3_cost_down = hotel3.order_by('-cost')
        hotel3_cost_down_1to10 = hotel3_cost_down[0:10]
        hotel3_cost_down_11to20 = hotel3_cost_down[11:21]
        hotel3_cost_down_21to30 = hotel3_cost_down[22:32]
        hotel3_cost_down_31to40 = hotel3_cost_down[33:43]
        hotel3_cost_down_41to50 = hotel3_cost_down[44:54]
        hotel3_cost_down_51to60 = hotel3_cost_down[55:65]
        hotel3_cost_down_61to70 = hotel3_cost_down[66:76]
        hotel3_cost_down_71to80 = hotel3_cost_down[77:87]
        hotel3_cost_down_81to90 = hotel3_cost_down[88:98]
        hotel3_cost_down_91to100 = hotel3_cost_down[99:109]
        hotel3_cost_down_101to110 = hotel3_cost_down[110:120]
        hotel3_cost_down_111to120 = hotel3_cost_down[121:131]
        hotel3_cost_down_121to130 = hotel3_cost_down[132:142]
        hotel3_cost_down_131to140 = hotel3_cost_down[143:153]
        hotel3_cost_down_141to150 = hotel3_cost_down[154:164]
        hotel3_cost_down_151to160 = hotel3_cost_down[165:175]
        hotel3_cost_down_161to170 = hotel3_cost_down[176:186]
        hotel3_cost_down_171to180 = hotel3_cost_down[187:197]
        hotel3_cost_down_181to190 = hotel3_cost_down[198:208]
        hotel3_cost_down_191to200 = hotel3_cost_down[209:219]
        hotel3_cost_down_201to210 = hotel3_cost_down[220:230]
        hotel3_cost_down_211to220 = hotel3_cost_down[231:241]

        hotel3_rating_down = hotel3.order_by('-rating')
        hotel3_rating_down_1to10 = hotel3_rating_down[0:10]
        hotel3_rating_down_11to20 = hotel3_rating_down[11:21]
        hotel3_rating_down_21to30 = hotel3_rating_down[22:32]
        hotel3_rating_down_31to40 = hotel3_rating_down[33:43]
        hotel3_rating_down_41to50 = hotel3_rating_down[44:54]
        hotel3_rating_down_51to60 = hotel3_rating_down[55:65]
        hotel3_rating_down_61to70 = hotel3_rating_down[66:76]
        hotel3_rating_down_71to80 = hotel3_rating_down[77:87]
        hotel3_rating_down_81to90 = hotel3_rating_down[88:98]
        hotel3_rating_down_91to100 = hotel3_rating_down[99:109]
        hotel3_rating_down_101to110 = hotel3_rating_down[110:120]
        hotel3_rating_down_111to120 = hotel3_rating_down[121:131]
        hotel3_rating_down_121to130 = hotel3_rating_down[132:142]
        hotel3_rating_down_131to140 = hotel3_rating_down[143:153]
        hotel3_rating_down_141to150 = hotel3_rating_down[154:164]
        hotel3_rating_down_151to160 = hotel3_rating_down[165:175]
        hotel3_rating_down_161to170 = hotel3_rating_down[176:186]
        hotel3_rating_down_171to180 = hotel3_rating_down[187:197]
        hotel3_rating_down_181to190 = hotel3_rating_down[198:208]
        hotel3_rating_down_191to200 = hotel3_rating_down[209:219]
        hotel3_rating_down_201to210 = hotel3_rating_down[220:230]
        hotel3_rating_down_211to220 = hotel3_rating_down[231:241]

        hotel3_distance_up = hotel3.order_by('distance')
        hotel3_distance_up_1to10 = hotel3_distance_up[0:10]
        hotel3_distance_up_11to20 = hotel3_distance_up[11:21]
        hotel3_distance_up_21to30 = hotel3_distance_up[22:32]
        hotel3_distance_up_31to40 = hotel3_distance_up[33:43]
        hotel3_distance_up_41to50 = hotel3_distance_up[44:54]
        hotel3_distance_up_51to60 = hotel3_distance_up[55:65]
        hotel3_distance_up_61to70 = hotel3_distance_up[66:76]
        hotel3_distance_up_71to80 = hotel3_distance_up[77:87]
        hotel3_distance_up_81to90 = hotel3_distance_up[88:98]
        hotel3_distance_up_91to100 = hotel3_distance_up[99:109]
        hotel3_distance_up_101to110 = hotel3_distance_up[110:120]
        hotel3_distance_up_111to120 = hotel3_distance_up[121:131]
        hotel3_distance_up_121to130 = hotel3_distance_up[132:142]
        hotel3_distance_up_131to140 = hotel3_distance_up[143:153]
        hotel3_distance_up_141to150 = hotel3_distance_up[154:164]
        hotel3_distance_up_151to160 = hotel3_distance_up[165:175]
        hotel3_distance_up_161to170 = hotel3_distance_up[176:186]
        hotel3_distance_up_171to180 = hotel3_distance_up[187:197]
        hotel3_distance_up_181to190 = hotel3_distance_up[198:208]
        hotel3_distance_up_191to200 = hotel3_distance_up[209:219]
        hotel3_distance_up_201to210 = hotel3_distance_up[220:230]
        hotel3_distance_up_211to220 = hotel3_distance_up[231:241]

        hotel3_kind_down = hotel3.order_by('-kind')
        hotel3_kind_down_1to10 = hotel3_kind_down[0:10]
        hotel3_kind_down_11to20 = hotel3_kind_down[11:21]
        hotel3_kind_down_21to30 = hotel3_kind_down[22:32]
        hotel3_kind_down_31to40 = hotel3_kind_down[33:43]
        hotel3_kind_down_41to50 = hotel3_kind_down[44:54]
        hotel3_kind_down_51to60 = hotel3_kind_down[55:65]
        hotel3_kind_down_61to70 = hotel3_kind_down[66:76]
        hotel3_kind_down_71to80 = hotel3_kind_down[77:87]
        hotel3_kind_down_81to90 = hotel3_kind_down[88:98]
        hotel3_kind_down_91to100 = hotel3_kind_down[99:109]
        hotel3_kind_down_101to110 = hotel3_kind_down[110:120]
        hotel3_kind_down_111to120 = hotel3_kind_down[121:131]
        hotel3_kind_down_121to130 = hotel3_kind_down[132:142]
        hotel3_kind_down_131to140 = hotel3_kind_down[143:153]
        hotel3_kind_down_141to150 = hotel3_kind_down[154:164]
        hotel3_kind_down_151to160 = hotel3_kind_down[165:175]
        hotel3_kind_down_161to170 = hotel3_kind_down[176:186]
        hotel3_kind_down_171to180 = hotel3_kind_down[187:197]
        hotel3_kind_down_181to190 = hotel3_kind_down[198:208]
        hotel3_kind_down_191to200 = hotel3_kind_down[209:219]
        hotel3_kind_down_201to210 = hotel3_kind_down[220:230]
        hotel3_kind_down_211to220 = hotel3_kind_down[231:241]

        hotel3_clean_down = hotel3.order_by('-clean')
        hotel3_clean_down_1to10 = hotel3_clean_down[0:10]
        hotel3_clean_down_11to20 = hotel3_clean_down[11:21]
        hotel3_clean_down_21to30 = hotel3_clean_down[22:32]
        hotel3_clean_down_31to40 = hotel3_clean_down[33:43]
        hotel3_clean_down_41to50 = hotel3_clean_down[44:54]
        hotel3_clean_down_51to60 = hotel3_clean_down[55:65]
        hotel3_clean_down_61to70 = hotel3_clean_down[66:76]
        hotel3_clean_down_71to80 = hotel3_clean_down[77:87]
        hotel3_clean_down_81to90 = hotel3_clean_down[88:98]
        hotel3_clean_down_91to100 = hotel3_clean_down[99:109]
        hotel3_clean_down_101to110 = hotel3_clean_down[110:120]
        hotel3_clean_down_111to120 = hotel3_clean_down[121:131]
        hotel3_clean_down_121to130 = hotel3_clean_down[132:142]
        hotel3_clean_down_131to140 = hotel3_clean_down[143:153]
        hotel3_clean_down_141to150 = hotel3_clean_down[154:164]
        hotel3_clean_down_151to160 = hotel3_clean_down[165:175]
        hotel3_clean_down_161to170 = hotel3_clean_down[176:186]
        hotel3_clean_down_171to180 = hotel3_clean_down[187:197]
        hotel3_clean_down_181to190 = hotel3_clean_down[198:208]
        hotel3_clean_down_191to200 = hotel3_clean_down[209:219]
        hotel3_clean_down_201to210 = hotel3_clean_down[220:230]
        hotel3_clean_down_211to220 = hotel3_clean_down[231:241]

        hotel3_conv_down = hotel3.order_by('-conv')
        hotel3_conv_down_1to10 = hotel3_conv_down[0:10]
        hotel3_conv_down_11to20 = hotel3_conv_down[11:21]
        hotel3_conv_down_21to30 = hotel3_conv_down[22:32]
        hotel3_conv_down_31to40 = hotel3_conv_down[33:43]
        hotel3_conv_down_41to50 = hotel3_conv_down[44:54]
        hotel3_conv_down_51to60 = hotel3_conv_down[55:65]
        hotel3_conv_down_61to70 = hotel3_conv_down[66:76]
        hotel3_conv_down_71to80 = hotel3_conv_down[77:87]
        hotel3_conv_down_81to90 = hotel3_conv_down[88:98]
        hotel3_conv_down_91to100 = hotel3_conv_down[99:109]
        hotel3_conv_down_101to110 = hotel3_conv_down[110:120]
        hotel3_conv_down_111to120 = hotel3_conv_down[121:131]
        hotel3_conv_down_121to130 = hotel3_conv_down[132:142]
        hotel3_conv_down_131to140 = hotel3_conv_down[143:153]
        hotel3_conv_down_141to150 = hotel3_conv_down[154:164]
        hotel3_conv_down_151to160 = hotel3_conv_down[165:175]
        hotel3_conv_down_161to170 = hotel3_conv_down[176:186]
        hotel3_conv_down_171to180 = hotel3_conv_down[187:197]
        hotel3_conv_down_181to190 = hotel3_conv_down[198:208]
        hotel3_conv_down_191to200 = hotel3_conv_down[209:219]
        hotel3_conv_down_201to210 = hotel3_conv_down[220:230]
        hotel3_conv_down_211to220 = hotel3_conv_down[231:241]

        hotel3_hotel = Hotel.objects.filter(place=result[2],
                                            classfication='호텔')
        hotel3_hotel_1to10 = hotel3_hotel[0:10]
        hotel3_hotel_11to20 = hotel3_hotel[11:21]
        hotel3_hotel_21to30 = hotel3_hotel[22:32]
        hotel3_hotel_31to40 = hotel3_hotel[33:43]
        hotel3_hotel_41to50 = hotel3_hotel[44:54]
        hotel3_hotel_51to60 = hotel3_hotel[55:65]
        hotel3_hotel_61to70 = hotel3_hotel[66:76]
        hotel3_hotel_71to80 = hotel3_hotel[77:87]
        hotel3_hotel_81to90 = hotel3_hotel[88:98]
        hotel3_hotel_91to100 = hotel3_hotel[99:109]
        hotel3_hotel_101to110 = hotel3_hotel[110:120]
        hotel3_hotel_111to120 = hotel3_hotel[121:131]
        hotel3_hotel_121to130 = hotel3_hotel[132:142]
        hotel3_hotel_131to140 = hotel3_hotel[143:153]
        hotel3_hotel_141to150 = hotel3_hotel[154:164]
        hotel3_hotel_151to160 = hotel3_hotel[165:175]
        hotel3_hotel_161to170 = hotel3_hotel[176:186]
        hotel3_hotel_171to180 = hotel3_hotel[187:197]
        hotel3_hotel_181to190 = hotel3_hotel[198:208]
        hotel3_hotel_191to200 = hotel3_hotel[209:219]
        hotel3_hotel_201to210 = hotel3_hotel[220:230]
        hotel3_hotel_211to220 = hotel3_hotel[231:241]

        hotel3_hostel = Hotel.objects.filter(place=result[2],
                                             classfication='호스텔')
        hotel3_hostel_1to10 = hotel3_hostel[0:10]
        hotel3_hostel_11to20 = hotel3_hostel[11:21]
        hotel3_hostel_21to30 = hotel3_hostel[22:32]
        hotel3_hostel_31to40 = hotel3_hostel[33:43]
        hotel3_hostel_41to50 = hotel3_hostel[44:54]
        hotel3_hostel_51to60 = hotel3_hostel[55:65]
        hotel3_hostel_61to70 = hotel3_hostel[66:76]
        hotel3_hostel_71to80 = hotel3_hostel[77:87]
        hotel3_hostel_81to90 = hotel3_hostel[88:98]
        hotel3_hostel_91to100 = hotel3_hostel[99:109]
        hotel3_hostel_101to110 = hotel3_hostel[110:120]
        hotel3_hostel_111to120 = hotel3_hostel[121:131]
        hotel3_hostel_121to130 = hotel3_hostel[132:142]
        hotel3_hostel_131to140 = hotel3_hostel[143:153]
        hotel3_hostel_141to150 = hotel3_hostel[154:164]
        hotel3_hostel_151to160 = hotel3_hostel[165:175]
        hotel3_hostel_161to170 = hotel3_hostel[176:186]
        hotel3_hostel_171to180 = hotel3_hostel[187:197]
        hotel3_hostel_181to190 = hotel3_hostel[198:208]
        hotel3_hostel_191to200 = hotel3_hostel[209:219]
        hotel3_hostel_201to210 = hotel3_hostel[220:230]
        hotel3_hostel_211to220 = hotel3_hostel[231:241]

        hotel3_guest = Hotel.objects.filter(place=result[2],
                                            classfication='게스트하우스')
        hotel3_guest_1to10 = hotel3_guest[0:10]
        hotel3_guest_11to20 = hotel3_guest[11:21]
        hotel3_guest_21to30 = hotel3_guest[22:32]
        hotel3_guest_31to40 = hotel3_guest[33:43]
        hotel3_guest_41to50 = hotel3_guest[44:54]
        hotel3_guest_51to60 = hotel3_guest[55:65]
        hotel3_guest_61to70 = hotel3_guest[66:76]
        hotel3_guest_71to80 = hotel3_guest[77:87]
        hotel3_guest_81to90 = hotel3_guest[88:98]
        hotel3_guest_91to100 = hotel3_guest[99:109]
        hotel3_guest_101to110 = hotel3_guest[110:120]
        hotel3_guest_111to120 = hotel3_guest[121:131]
        hotel3_guest_121to130 = hotel3_guest[132:142]
        hotel3_guest_131to140 = hotel3_guest[143:153]
        hotel3_guest_141to150 = hotel3_guest[154:164]
        hotel3_guest_151to160 = hotel3_guest[165:175]
        hotel3_guest_161to170 = hotel3_guest[176:186]
        hotel3_guest_171to180 = hotel3_guest[187:197]
        hotel3_guest_181to190 = hotel3_guest[198:208]
        hotel3_guest_191to200 = hotel3_guest[209:219]
        hotel3_guest_201to210 = hotel3_guest[220:230]
        hotel3_guest_211to220 = hotel3_guest[231:241]

        hotel3_apartment = Hotel.objects.filter(place=result[2],
                                                classfication='아파트')
        hotel3_apartment_1to10 = hotel3_apartment[0:10]
        hotel3_apartment_11to20 = hotel3_apartment[11:21]
        hotel3_apartment_21to30 = hotel3_apartment[22:32]
        hotel3_apartment_31to40 = hotel3_apartment[33:43]
        hotel3_apartment_41to50 = hotel3_apartment[44:54]
        hotel3_apartment_51to60 = hotel3_apartment[55:65]
        hotel3_apartment_61to70 = hotel3_apartment[66:76]
        hotel3_apartment_71to80 = hotel3_apartment[77:87]
        hotel3_apartment_81to90 = hotel3_apartment[88:98]
        hotel3_apartment_91to100 = hotel3_apartment[99:109]
        hotel3_apartment_101to110 = hotel3_apartment[110:120]
        hotel3_apartment_111to120 = hotel3_apartment[121:131]
        hotel3_apartment_121to130 = hotel3_apartment[132:142]
        hotel3_apartment_131to140 = hotel3_apartment[143:153]
        hotel3_apartment_141to150 = hotel3_apartment[154:164]
        hotel3_apartment_151to160 = hotel3_apartment[165:175]
        hotel3_apartment_161to170 = hotel3_apartment[176:186]
        hotel3_apartment_171to180 = hotel3_apartment[187:197]
        hotel3_apartment_181to190 = hotel3_apartment[198:208]
        hotel3_apartment_191to200 = hotel3_apartment[209:219]
        hotel3_apartment_201to210 = hotel3_apartment[220:230]
        hotel3_apartment_211to220 = hotel3_apartment[231:241]

        hotel3_apartmenthotel = Hotel.objects.filter(place=result[2],
                                                     classfication='아파트호텔')
        hotel3_apartmenthotel_1to10 = hotel3_apartmenthotel[0:10]
        hotel3_apartmenthotel_11to20 = hotel3_apartmenthotel[11:21]
        hotel3_apartmenthotel_21to30 = hotel3_apartmenthotel[22:32]
        hotel3_apartmenthotel_31to40 = hotel3_apartmenthotel[33:43]
        hotel3_apartmenthotel_41to50 = hotel3_apartmenthotel[44:54]
        hotel3_apartmenthotel_51to60 = hotel3_apartmenthotel[55:65]
        hotel3_apartmenthotel_61to70 = hotel3_apartmenthotel[66:76]
        hotel3_apartmenthotel_71to80 = hotel3_apartmenthotel[77:87]
        hotel3_apartmenthotel_81to90 = hotel3_apartmenthotel[88:98]
        hotel3_apartmenthotel_91to100 = hotel3_apartmenthotel[99:109]
        hotel3_apartmenthotel_101to110 = hotel3_apartmenthotel[110:120]
        hotel3_apartmenthotel_111to120 = hotel3_apartmenthotel[121:131]
        hotel3_apartmenthotel_121to130 = hotel3_apartmenthotel[132:142]
        hotel3_apartmenthotel_131to140 = hotel3_apartmenthotel[143:153]
        hotel3_apartmenthotel_141to150 = hotel3_apartmenthotel[154:164]
        hotel3_apartmenthotel_151to160 = hotel3_apartmenthotel[165:175]
        hotel3_apartmenthotel_161to170 = hotel3_apartmenthotel[176:186]
        hotel3_apartmenthotel_171to180 = hotel3_apartmenthotel[187:197]
        hotel3_apartmenthotel_181to190 = hotel3_apartmenthotel[198:208]
        hotel3_apartmenthotel_191to200 = hotel3_apartmenthotel[209:219]
        hotel3_apartmenthotel_201to210 = hotel3_apartmenthotel[220:230]
        hotel3_apartmenthotel_211to220 = hotel3_apartmenthotel[231:241]

        hotel3_motel = Hotel.objects.filter(place=result[2],
                                            classfication='모텔')
        hotel3_motel_1to10 = hotel3_motel[0:10]
        hotel3_motel_11to20 = hotel3_motel[11:21]
        hotel3_motel_21to30 = hotel3_motel[22:32]
        hotel3_motel_31to40 = hotel3_motel[33:43]
        hotel3_motel_41to50 = hotel3_motel[44:54]
        hotel3_motel_51to60 = hotel3_motel[55:65]
        hotel3_motel_61to70 = hotel3_motel[66:76]
        hotel3_motel_71to80 = hotel3_motel[77:87]
        hotel3_motel_81to90 = hotel3_motel[88:98]
        hotel3_motel_91to100 = hotel3_motel[99:109]
        hotel3_motel_101to110 = hotel3_motel[110:120]
        hotel3_motel_111to120 = hotel3_motel[121:131]
        hotel3_motel_121to130 = hotel3_motel[132:142]
        hotel3_motel_131to140 = hotel3_motel[143:153]
        hotel3_motel_141to150 = hotel3_motel[154:164]
        hotel3_motel_151to160 = hotel3_motel[165:175]
        hotel3_motel_161to170 = hotel3_motel[176:186]
        hotel3_motel_171to180 = hotel3_motel[187:197]
        hotel3_motel_181to190 = hotel3_motel[198:208]
        hotel3_motel_191to200 = hotel3_motel[209:219]
        hotel3_motel_201to210 = hotel3_motel[220:230]
        hotel3_motel_211to220 = hotel3_motel[231:241]

        hotel3_pension = Hotel.objects.filter(place=result[2],
                                              classfication='펜션')
        hotel3_pension_1to10 = hotel3_pension[0:10]
        hotel3_pension_11to20 = hotel3_pension[11:21]
        hotel3_pension_21to30 = hotel3_pension[22:32]
        hotel3_pension_31to40 = hotel3_pension[33:43]
        hotel3_pension_41to50 = hotel3_pension[44:54]
        hotel3_pension_51to60 = hotel3_pension[55:65]
        hotel3_pension_61to70 = hotel3_pension[66:76]
        hotel3_pension_71to80 = hotel3_pension[77:87]
        hotel3_pension_81to90 = hotel3_pension[88:98]
        hotel3_pension_91to100 = hotel3_pension[99:109]
        hotel3_pension_101to110 = hotel3_pension[110:120]
        hotel3_pension_111to120 = hotel3_pension[121:131]
        hotel3_pension_121to130 = hotel3_pension[132:142]
        hotel3_pension_131to140 = hotel3_pension[143:153]
        hotel3_pension_141to150 = hotel3_pension[154:164]
        hotel3_pension_151to160 = hotel3_pension[165:175]
        hotel3_pension_161to170 = hotel3_pension[176:186]
        hotel3_pension_171to180 = hotel3_pension[187:197]
        hotel3_pension_181to190 = hotel3_pension[198:208]
        hotel3_pension_191to200 = hotel3_pension[209:219]
        hotel3_pension_201to210 = hotel3_pension[220:230]
        hotel3_pension_211to220 = hotel3_pension[231:241]

        hotel3_resort = Hotel.objects.filter(place=result[2],
                                             classfication='리조트')
        hotel3_resort_1to10 = hotel3_resort[0:10]
        hotel3_resort_11to20 = hotel3_resort[11:21]
        hotel3_resort_21to30 = hotel3_resort[22:32]
        hotel3_resort_31to40 = hotel3_resort[33:43]
        hotel3_resort_41to50 = hotel3_resort[44:54]
        hotel3_resort_51to60 = hotel3_resort[55:65]
        hotel3_resort_61to70 = hotel3_resort[66:76]
        hotel3_resort_71to80 = hotel3_resort[77:87]
        hotel3_resort_81to90 = hotel3_resort[88:98]
        hotel3_resort_91to100 = hotel3_resort[99:109]
        hotel3_resort_101to110 = hotel3_resort[110:120]
        hotel3_resort_111to120 = hotel3_resort[121:131]
        hotel3_resort_121to130 = hotel3_resort[132:142]
        hotel3_resort_131to140 = hotel3_resort[143:153]
        hotel3_resort_141to150 = hotel3_resort[154:164]
        hotel3_resort_151to160 = hotel3_resort[165:175]
        hotel3_resort_161to170 = hotel3_resort[176:186]
        hotel3_resort_171to180 = hotel3_resort[187:197]
        hotel3_resort_181to190 = hotel3_resort[198:208]
        hotel3_resort_191to200 = hotel3_resort[209:219]
        hotel3_resort_201to210 = hotel3_resort[220:230]
        hotel3_resort_211to220 = hotel3_resort[231:241]

        hotel3_badandbreakfast = Hotel.objects.filter(place=result[2],
                                                      classfication='베드앤브렉퍼스트')
        hotel3_badandbreakfast_1to10 = hotel3_badandbreakfast[0:10]
        hotel3_badandbreakfast_11to20 = hotel3_badandbreakfast[11:21]
        hotel3_badandbreakfast_21to30 = hotel3_badandbreakfast[22:32]
        hotel3_badandbreakfast_31to40 = hotel3_badandbreakfast[33:43]
        hotel3_badandbreakfast_41to50 = hotel3_badandbreakfast[44:54]
        hotel3_badandbreakfast_51to60 = hotel3_badandbreakfast[55:65]
        hotel3_badandbreakfast_61to70 = hotel3_badandbreakfast[66:76]
        hotel3_badandbreakfast_71to80 = hotel3_badandbreakfast[77:87]
        hotel3_badandbreakfast_81to90 = hotel3_badandbreakfast[88:98]
        hotel3_badandbreakfast_91to100 = hotel3_badandbreakfast[99:109]
        hotel3_badandbreakfast_101to110 = hotel3_badandbreakfast[110:120]
        hotel3_badandbreakfast_111to120 = hotel3_badandbreakfast[121:131]
        hotel3_badandbreakfast_121to130 = hotel3_badandbreakfast[132:142]
        hotel3_badandbreakfast_131to140 = hotel3_badandbreakfast[143:153]
        hotel3_badandbreakfast_141to150 = hotel3_badandbreakfast[154:164]
        hotel3_badandbreakfast_151to160 = hotel3_badandbreakfast[165:175]
        hotel3_badandbreakfast_161to170 = hotel3_badandbreakfast[176:186]
        hotel3_badandbreakfast_171to180 = hotel3_badandbreakfast[187:197]
        hotel3_badandbreakfast_181to190 = hotel3_badandbreakfast[198:208]
        hotel3_badandbreakfast_191to200 = hotel3_badandbreakfast[209:219]
        hotel3_badandbreakfast_201to210 = hotel3_badandbreakfast[220:230]
        hotel3_badandbreakfast_211to220 = hotel3_badandbreakfast[231:241]

        hotel3_homestay = Hotel.objects.filter(place=result[2],
                                               classfication='홈스테이')
        hotel3_homestay_1to10 = hotel3_homestay[0:10]
        hotel3_homestay_11to20 = hotel3_homestay[11:21]
        hotel3_homestay_21to30 = hotel3_homestay[22:32]
        hotel3_homestay_31to40 = hotel3_homestay[33:43]
        hotel3_homestay_41to50 = hotel3_homestay[44:54]
        hotel3_homestay_51to60 = hotel3_homestay[55:65]
        hotel3_homestay_61to70 = hotel3_homestay[66:76]
        hotel3_homestay_71to80 = hotel3_homestay[77:87]
        hotel3_homestay_81to90 = hotel3_homestay[88:98]
        hotel3_homestay_91to100 = hotel3_homestay[99:109]
        hotel3_homestay_101to110 = hotel3_homestay[110:120]
        hotel3_homestay_111to120 = hotel3_homestay[121:131]
        hotel3_homestay_121to130 = hotel3_homestay[132:142]
        hotel3_homestay_131to140 = hotel3_homestay[143:153]
        hotel3_homestay_141to150 = hotel3_homestay[154:164]
        hotel3_homestay_151to160 = hotel3_homestay[165:175]
        hotel3_homestay_161to170 = hotel3_homestay[176:186]
        hotel3_homestay_171to180 = hotel3_homestay[187:197]
        hotel3_homestay_181to190 = hotel3_homestay[198:208]
        hotel3_homestay_191to200 = hotel3_homestay[209:219]
        hotel3_homestay_201to210 = hotel3_homestay[220:230]
        hotel3_homestay_211to220 = hotel3_homestay[231:241]

        hotel3_lodge = Hotel.objects.filter(place=result[2],
                                            classfication='롯지')
        hotel3_lodge_1to10 = hotel3_lodge[0:10]
        hotel3_lodge_11to20 = hotel3_lodge[11:21]
        hotel3_lodge_21to30 = hotel3_lodge[22:32]
        hotel3_lodge_31to40 = hotel3_lodge[33:43]
        hotel3_lodge_41to50 = hotel3_lodge[44:54]
        hotel3_lodge_51to60 = hotel3_lodge[55:65]
        hotel3_lodge_61to70 = hotel3_lodge[66:76]
        hotel3_lodge_71to80 = hotel3_lodge[77:87]
        hotel3_lodge_81to90 = hotel3_lodge[88:98]
        hotel3_lodge_91to100 = hotel3_lodge[99:109]
        hotel3_lodge_101to110 = hotel3_lodge[110:120]
        hotel3_lodge_111to120 = hotel3_lodge[121:131]
        hotel3_lodge_121to130 = hotel3_lodge[132:142]
        hotel3_lodge_131to140 = hotel3_lodge[143:153]
        hotel3_lodge_141to150 = hotel3_lodge[154:164]
        hotel3_lodge_151to160 = hotel3_lodge[165:175]
        hotel3_lodge_161to170 = hotel3_lodge[176:186]
        hotel3_lodge_171to180 = hotel3_lodge[187:197]
        hotel3_lodge_181to190 = hotel3_lodge[198:208]
        hotel3_lodge_191to200 = hotel3_lodge[209:219]
        hotel3_lodge_201to210 = hotel3_lodge[220:230]
        hotel3_lodge_211to220 = hotel3_lodge[231:241]

        hotel3_countryhouse = Hotel.objects.filter(place=result[2],
                                                   classfication='컨트리하우스')
        hotel3_countryhouse_1to10 = hotel3_countryhouse[0:10]
        hotel3_countryhouse_11to20 = hotel3_countryhouse[11:21]
        hotel3_countryhouse_21to30 = hotel3_countryhouse[22:32]
        hotel3_countryhouse_31to40 = hotel3_countryhouse[33:43]
        hotel3_countryhouse_41to50 = hotel3_countryhouse[44:54]
        hotel3_countryhouse_51to60 = hotel3_countryhouse[55:65]
        hotel3_countryhouse_61to70 = hotel3_countryhouse[66:76]
        hotel3_countryhouse_71to80 = hotel3_countryhouse[77:87]
        hotel3_countryhouse_81to90 = hotel3_countryhouse[88:98]
        hotel3_countryhouse_91to100 = hotel3_countryhouse[99:109]
        hotel3_countryhouse_101to110 = hotel3_countryhouse[110:120]
        hotel3_countryhouse_111to120 = hotel3_countryhouse[121:131]
        hotel3_countryhouse_121to130 = hotel3_countryhouse[132:142]
        hotel3_countryhouse_131to140 = hotel3_countryhouse[143:153]
        hotel3_countryhouse_141to150 = hotel3_countryhouse[154:164]
        hotel3_countryhouse_151to160 = hotel3_countryhouse[165:175]
        hotel3_countryhouse_161to170 = hotel3_countryhouse[176:186]
        hotel3_countryhouse_171to180 = hotel3_countryhouse[187:197]
        hotel3_countryhouse_181to190 = hotel3_countryhouse[198:208]
        hotel3_countryhouse_191to200 = hotel3_countryhouse[209:219]
        hotel3_countryhouse_201to210 = hotel3_countryhouse[220:230]
        hotel3_countryhouse_211to220 = hotel3_countryhouse[231:241]

        hotel3_inn = Hotel.objects.filter(place=result[2], classfication='여관')
        hotel3_inn_1to10 = hotel3_inn[0:10]
        hotel3_inn_11to20 = hotel3_inn[11:21]
        hotel3_inn_21to30 = hotel3_inn[22:32]
        hotel3_inn_31to40 = hotel3_inn[33:43]
        hotel3_inn_41to50 = hotel3_inn[44:54]
        hotel3_inn_51to60 = hotel3_inn[55:65]
        hotel3_inn_61to70 = hotel3_inn[66:76]
        hotel3_inn_71to80 = hotel3_inn[77:87]
        hotel3_inn_81to90 = hotel3_inn[88:98]
        hotel3_inn_91to100 = hotel3_inn[99:109]
        hotel3_inn_101to110 = hotel3_inn[110:120]
        hotel3_inn_111to120 = hotel3_inn[121:131]
        hotel3_inn_121to130 = hotel3_inn[132:142]
        hotel3_inn_131to140 = hotel3_inn[143:153]
        hotel3_inn_141to150 = hotel3_inn[154:164]
        hotel3_inn_151to160 = hotel3_inn[165:175]
        hotel3_inn_161to170 = hotel3_inn[176:186]
        hotel3_inn_171to180 = hotel3_inn[187:197]
        hotel3_inn_181to190 = hotel3_inn[198:208]
        hotel3_inn_191to200 = hotel3_inn[209:219]
        hotel3_inn_201to210 = hotel3_inn[220:230]
        hotel3_inn_211to220 = hotel3_inn[231:241]

        hotel3_villa = Hotel.objects.filter(place=result[2],
                                            classfication='빌라')
        hotel3_villa_1to10 = hotel3_villa[0:10]
        hotel3_villa_11to20 = hotel3_villa[11:21]
        hotel3_villa_21to30 = hotel3_villa[22:32]
        hotel3_villa_31to40 = hotel3_villa[33:43]
        hotel3_villa_41to50 = hotel3_villa[44:54]
        hotel3_villa_51to60 = hotel3_villa[55:65]
        hotel3_villa_61to70 = hotel3_villa[66:76]
        hotel3_villa_71to80 = hotel3_villa[77:87]
        hotel3_villa_81to90 = hotel3_villa[88:98]
        hotel3_villa_91to100 = hotel3_villa[99:109]
        hotel3_villa_101to110 = hotel3_villa[110:120]
        hotel3_villa_111to120 = hotel3_villa[121:131]
        hotel3_villa_121to130 = hotel3_villa[132:142]
        hotel3_villa_131to140 = hotel3_villa[143:153]
        hotel3_villa_141to150 = hotel3_villa[154:164]
        hotel3_villa_151to160 = hotel3_villa[165:175]
        hotel3_villa_161to170 = hotel3_villa[176:186]
        hotel3_villa_171to180 = hotel3_villa[187:197]
        hotel3_villa_181to190 = hotel3_villa[198:208]
        hotel3_villa_191to200 = hotel3_villa[209:219]
        hotel3_villa_201to210 = hotel3_villa[220:230]
        hotel3_villa_211to220 = hotel3_villa[231:241]

        hotel3_camping = Hotel.objects.filter(place=result[2],
                                              classfication='캠핑장')
        hotel3_camping_1to10 = hotel3_camping[0:10]
        hotel3_camping_11to20 = hotel3_camping[11:21]
        hotel3_camping_21to30 = hotel3_camping[22:32]
        hotel3_camping_31to40 = hotel3_camping[33:43]
        hotel3_camping_41to50 = hotel3_camping[44:54]
        hotel3_camping_51to60 = hotel3_camping[55:65]
        hotel3_camping_61to70 = hotel3_camping[66:76]
        hotel3_camping_71to80 = hotel3_camping[77:87]
        hotel3_camping_81to90 = hotel3_camping[88:98]
        hotel3_camping_91to100 = hotel3_camping[99:109]
        hotel3_camping_101to110 = hotel3_camping[110:120]
        hotel3_camping_111to120 = hotel3_camping[121:131]
        hotel3_camping_121to130 = hotel3_camping[132:142]
        hotel3_camping_131to140 = hotel3_camping[143:153]
        hotel3_camping_141to150 = hotel3_camping[154:164]
        hotel3_camping_151to160 = hotel3_camping[165:175]
        hotel3_camping_161to170 = hotel3_camping[176:186]
        hotel3_camping_171to180 = hotel3_camping[187:197]
        hotel3_camping_181to190 = hotel3_camping[198:208]
        hotel3_camping_191to200 = hotel3_camping[209:219]
        hotel3_camping_201to210 = hotel3_camping[220:230]
        hotel3_camping_211to220 = hotel3_camping[231:241]

        # restaurant1 = Restaurant.objects.filter(place=result[0])
        # restaurant2 = Restaurant.objects.filter(place=result[1])
        # restaurant3 = Restaurant.objects.filter(place=result[2])
        # restaurant4 = Restaurant.objects.filter(place=result[3])
        # restaurant5 = Restaurant.objects.filter(place=result[4])
        # 관광지, 식당 이름을 뽑는다.
        # 찜 목록에서는 어떤 관광지에 어떤 식당이다.

        return render(
            request,
            'beer/ver1_result.html',
            {
                'login_session': login_session,
                'result': result,
                'hotels1': hotel1,
                'hotel1_1to10': hotel1_1to10,
                'hotel1_11to20': hotel1_11to20,
                'hotel1_21to30': hotel1_21to30,
                'hotel1_31to40': hotel1_31to40,
                'hotel1_41to50': hotel1_41to50,
                'hotel1_51to60': hotel1_51to60,
                'hotel1_61to70': hotel1_61to70,
                'hotel1_71to80': hotel1_71to80,
                'hotel1_81to90': hotel1_81to90,
                'hotel1_91to100': hotel1_91to100,
                'hotel1_101to110': hotel1_101to110,
                'hotel1_111to120': hotel1_111to120,
                'hotel1_121to130': hotel1_121to130,
                'hotel1_131to140': hotel1_131to140,
                'hotel1_141to150': hotel1_141to150,
                'hotel1_151to160': hotel1_151to160,
                'hotel1_161to170': hotel1_161to170,
                'hotel1_171to180': hotel1_171to180,
                'hotel1_181to190': hotel1_181to190,
                'hotel1_191to200': hotel1_191to200,
                'hotel1_201to210': hotel1_201to210,
                'hotel1_211to220': hotel1_211to220,
                'hotel1_cost_up': hotel1_cost_up,
                'hotel1_cost_up_1to10': hotel1_cost_up_1to10,
                'hotel1_cost_up_11to20 ': hotel1_cost_up_11to20,
                'hotel1_cost_up_21to30 ': hotel1_cost_up_21to30,
                'hotel1_cost_up_31to40 ': hotel1_cost_up_31to40,
                'hotel1_cost_up_41to50 ': hotel1_cost_up_41to50,
                'hotel1_cost_up_51to60 ': hotel1_cost_up_51to60,
                'hotel1_cost_up_61to70 ': hotel1_cost_up_61to70,
                'hotel1_cost_up_71to80 ': hotel1_cost_up_71to80,
                'hotel1_cost_up_81to90 ': hotel1_cost_up_81to90,
                'hotel1_cost_up_91to100': hotel1_cost_up_91to100,
                'hotel1_cost_up_101to110': hotel1_cost_up_101to110,
                'hotel1_cost_up_111to120': hotel1_cost_up_111to120,
                'hotel1_cost_up_121to130': hotel1_cost_up_121to130,
                'hotel1_cost_up_131to140': hotel1_cost_up_131to140,
                'hotel1_cost_up_141to150': hotel1_cost_up_141to150,
                'hotel1_cost_up_151to160': hotel1_cost_up_151to160,
                'hotel1_cost_up_161to170': hotel1_cost_up_161to170,
                'hotel1_cost_up_171to180': hotel1_cost_up_171to180,
                'hotel1_cost_up_181to190': hotel1_cost_up_181to190,
                'hotel1_cost_up_191to200': hotel1_cost_up_191to200,
                'hotel1_cost_up_201to210': hotel1_cost_up_201to210,
                'hotel1_cost_up_211to220': hotel1_cost_up_211to220,
                'hotel1_cost_down': hotel1_cost_down,
                'hotel1_cost_down_1to10': hotel1_cost_down_1to10,
                'hotel1_cost_down_11to20 ': hotel1_cost_down_11to20,
                'hotel1_cost_down_21to30 ': hotel1_cost_down_21to30,
                'hotel1_cost_down_31to40 ': hotel1_cost_down_31to40,
                'hotel1_cost_down_41to50 ': hotel1_cost_down_41to50,
                'hotel1_cost_down_51to60 ': hotel1_cost_down_51to60,
                'hotel1_cost_down_61to70 ': hotel1_cost_down_61to70,
                'hotel1_cost_down_71to80 ': hotel1_cost_down_71to80,
                'hotel1_cost_down_81to90 ': hotel1_cost_down_81to90,
                'hotel1_cost_down_91to100': hotel1_cost_down_91to100,
                'hotel1_cost_down_101to110': hotel1_cost_down_101to110,
                'hotel1_cost_down_111to120': hotel1_cost_down_111to120,
                'hotel1_cost_down_121to130': hotel1_cost_down_121to130,
                'hotel1_cost_down_131to140': hotel1_cost_down_131to140,
                'hotel1_cost_down_141to150': hotel1_cost_down_141to150,
                'hotel1_cost_down_151to160': hotel1_cost_down_151to160,
                'hotel1_cost_down_161to170': hotel1_cost_down_161to170,
                'hotel1_cost_down_171to180': hotel1_cost_down_171to180,
                'hotel1_cost_down_181to190': hotel1_cost_down_181to190,
                'hotel1_cost_down_191to200': hotel1_cost_down_191to200,
                'hotel1_cost_down_201to210': hotel1_cost_down_201to210,
                'hotel1_cost_down_211to220': hotel1_cost_down_211to220,
                'hotel1_rating_down': hotel1_rating_down,
                'hotel1_rating_down_1to10': hotel1_rating_down_1to10,
                'hotel1_rating_down_11to20 ': hotel1_rating_down_11to20,
                'hotel1_rating_down_21to30 ': hotel1_rating_down_21to30,
                'hotel1_rating_down_31to40 ': hotel1_rating_down_31to40,
                'hotel1_rating_down_41to50 ': hotel1_rating_down_41to50,
                'hotel1_rating_down_51to60 ': hotel1_rating_down_51to60,
                'hotel1_rating_down_61to70 ': hotel1_rating_down_61to70,
                'hotel1_rating_down_71to80 ': hotel1_rating_down_71to80,
                'hotel1_rating_down_81to90 ': hotel1_rating_down_81to90,
                'hotel1_rating_down_91to100': hotel1_rating_down_91to100,
                'hotel1_rating_down_101to110': hotel1_rating_down_101to110,
                'hotel1_rating_down_111to120': hotel1_rating_down_111to120,
                'hotel1_rating_down_121to130': hotel1_rating_down_121to130,
                'hotel1_rating_down_131to140': hotel1_rating_down_131to140,
                'hotel1_rating_down_141to150': hotel1_rating_down_141to150,
                'hotel1_rating_down_151to160': hotel1_rating_down_151to160,
                'hotel1_rating_down_161to170': hotel1_rating_down_161to170,
                'hotel1_rating_down_171to180': hotel1_rating_down_171to180,
                'hotel1_rating_down_181to190': hotel1_rating_down_181to190,
                'hotel1_rating_down_191to200': hotel1_rating_down_191to200,
                'hotel1_rating_down_201to210': hotel1_rating_down_201to210,
                'hotel1_rating_down_211to220': hotel1_rating_down_211to220,
                'hotel1_distance_up': hotel1_distance_up,
                'hotel1_distance_up_1to10': hotel1_distance_up_1to10,
                'hotel1_distance_up_11to20 ': hotel1_distance_up_11to20,
                'hotel1_distance_up_21to30 ': hotel1_distance_up_21to30,
                'hotel1_distance_up_31to40 ': hotel1_distance_up_31to40,
                'hotel1_distance_up_41to50 ': hotel1_distance_up_41to50,
                'hotel1_distance_up_51to60 ': hotel1_distance_up_51to60,
                'hotel1_distance_up_61to70 ': hotel1_distance_up_61to70,
                'hotel1_distance_up_71to80 ': hotel1_distance_up_71to80,
                'hotel1_distance_up_81to90 ': hotel1_distance_up_81to90,
                'hotel1_distance_up_91to100': hotel1_distance_up_91to100,
                'hotel1_distance_up_101to110 ': hotel1_distance_up_101to110,
                'hotel1_distance_up_111to120 ': hotel1_distance_up_111to120,
                'hotel1_distance_up_121to130 ': hotel1_distance_up_121to130,
                'hotel1_distance_up_131to140 ': hotel1_distance_up_131to140,
                'hotel1_distance_up_141to150 ': hotel1_distance_up_141to150,
                'hotel1_distance_up_151to160 ': hotel1_distance_up_151to160,
                'hotel1_distance_up_161to170 ': hotel1_distance_up_161to170,
                'hotel1_distance_up_171to180 ': hotel1_distance_up_171to180,
                'hotel1_distance_up_181to190 ': hotel1_distance_up_181to190,
                'hotel1_distance_up_191to200 ': hotel1_distance_up_191to200,
                'hotel1_distance_up_201to210 ': hotel1_distance_up_201to210,
                'hotel1_distance_up_211to220': hotel1_distance_up_211to220,
                'hotel1_kind_down_1to10': hotel1_kind_down_1to10,
                'hotel1_kind_down_11to20 ': hotel1_kind_down_11to20,
                'hotel1_kind_down_21to30 ': hotel1_kind_down_21to30,
                'hotel1_kind_down_31to40 ': hotel1_kind_down_31to40,
                'hotel1_kind_down_41to50 ': hotel1_kind_down_41to50,
                'hotel1_kind_down_51to60 ': hotel1_kind_down_51to60,
                'hotel1_kind_down_61to70 ': hotel1_kind_down_61to70,
                'hotel1_kind_down_71to80 ': hotel1_kind_down_71to80,
                'hotel1_kind_down_81to90 ': hotel1_kind_down_81to90,
                'hotel1_kind_down_91to100': hotel1_kind_down_91to100,
                'hotel1_kind_down_101to110 ': hotel1_kind_down_101to110,
                'hotel1_kind_down_111to120 ': hotel1_kind_down_111to120,
                'hotel1_kind_down_121to130 ': hotel1_kind_down_121to130,
                'hotel1_kind_down_131to140 ': hotel1_kind_down_131to140,
                'hotel1_kind_down_141to150 ': hotel1_kind_down_141to150,
                'hotel1_kind_down_151to160 ': hotel1_kind_down_151to160,
                'hotel1_kind_down_161to170 ': hotel1_kind_down_161to170,
                'hotel1_kind_down_171to180 ': hotel1_kind_down_171to180,
                'hotel1_kind_down_181to190 ': hotel1_kind_down_181to190,
                'hotel1_kind_down_191to200 ': hotel1_kind_down_191to200,
                'hotel1_kind_down_201to210 ': hotel1_kind_down_201to210,
                'hotel1_kind_down_211to220': hotel1_kind_down_211to220,
                'hotel1_clean_down_1to10': hotel1_clean_down_1to10,
                'hotel1_clean_down_11to20 ': hotel1_clean_down_11to20,
                'hotel1_clean_down_21to30 ': hotel1_clean_down_21to30,
                'hotel1_clean_down_31to40 ': hotel1_clean_down_31to40,
                'hotel1_clean_down_41to50 ': hotel1_clean_down_41to50,
                'hotel1_clean_down_51to60 ': hotel1_clean_down_51to60,
                'hotel1_clean_down_61to70 ': hotel1_clean_down_61to70,
                'hotel1_clean_down_71to80 ': hotel1_clean_down_71to80,
                'hotel1_clean_down_81to90 ': hotel1_clean_down_81to90,
                'hotel1_clean_down_91to100': hotel1_clean_down_91to100,
                'hotel1_clean_down_101to110 ': hotel1_clean_down_101to110,
                'hotel1_clean_down_111to120 ': hotel1_clean_down_111to120,
                'hotel1_clean_down_121to130 ': hotel1_clean_down_121to130,
                'hotel1_clean_down_131to140 ': hotel1_clean_down_131to140,
                'hotel1_clean_down_141to150 ': hotel1_clean_down_141to150,
                'hotel1_clean_down_151to160 ': hotel1_clean_down_151to160,
                'hotel1_clean_down_161to170 ': hotel1_clean_down_161to170,
                'hotel1_clean_down_171to180 ': hotel1_clean_down_171to180,
                'hotel1_clean_down_181to190 ': hotel1_clean_down_181to190,
                'hotel1_clean_down_191to200 ': hotel1_clean_down_191to200,
                'hotel1_clean_down_201to210 ': hotel1_clean_down_201to210,
                'hotel1_clean_down_211to220': hotel1_clean_down_211to220,
                'hotel1_conv_down_1to10': hotel1_conv_down_1to10,
                'hotel1_conv_down_11to20 ': hotel1_conv_down_11to20,
                'hotel1_conv_down_21to30 ': hotel1_conv_down_21to30,
                'hotel1_conv_down_31to40 ': hotel1_conv_down_31to40,
                'hotel1_conv_down_41to50 ': hotel1_conv_down_41to50,
                'hotel1_conv_down_51to60 ': hotel1_conv_down_51to60,
                'hotel1_conv_down_61to70 ': hotel1_conv_down_61to70,
                'hotel1_conv_down_71to80 ': hotel1_conv_down_71to80,
                'hotel1_conv_down_81to90 ': hotel1_conv_down_81to90,
                'hotel1_conv_down_91to100': hotel1_conv_down_91to100,
                'hotel1_conv_down_101to110 ': hotel1_conv_down_101to110,
                'hotel1_conv_down_111to120 ': hotel1_conv_down_111to120,
                'hotel1_conv_down_121to130 ': hotel1_conv_down_121to130,
                'hotel1_conv_down_131to140 ': hotel1_conv_down_131to140,
                'hotel1_conv_down_141to150 ': hotel1_conv_down_141to150,
                'hotel1_conv_down_151to160 ': hotel1_conv_down_151to160,
                'hotel1_conv_down_161to170 ': hotel1_conv_down_161to170,
                'hotel1_conv_down_171to180 ': hotel1_conv_down_171to180,
                'hotel1_conv_down_181to190 ': hotel1_conv_down_181to190,
                'hotel1_conv_down_191to200 ': hotel1_conv_down_191to200,
                'hotel1_conv_down_201to210 ': hotel1_conv_down_201to210,
                'hotel1_conv_down_211to220': hotel1_conv_down_211to220,
                'hotel1_hotel': hotel1_hotel,
                'hotel1_hotel_1to10': hotel1_hotel_1to10,
                'hotel1_hotel_11to20 ': hotel1_hotel_11to20,
                'hotel1_hotel_21to30 ': hotel1_hotel_21to30,
                'hotel1_hotel_31to40 ': hotel1_hotel_31to40,
                'hotel1_hotel_41to50 ': hotel1_hotel_41to50,
                'hotel1_hotel_51to60 ': hotel1_hotel_51to60,
                'hotel1_hotel_61to70 ': hotel1_hotel_61to70,
                'hotel1_hotel_71to80 ': hotel1_hotel_71to80,
                'hotel1_hotel_81to90 ': hotel1_hotel_81to90,
                'hotel1_hotel_91to100': hotel1_hotel_91to100,
                'hotel1_hotel_101to110 ': hotel1_hotel_101to110,
                'hotel1_hotel_111to120 ': hotel1_hotel_111to120,
                'hotel1_hotel_121to130 ': hotel1_hotel_121to130,
                'hotel1_hotel_131to140 ': hotel1_hotel_131to140,
                'hotel1_hotel_141to150 ': hotel1_hotel_141to150,
                'hotel1_hotel_151to160 ': hotel1_hotel_151to160,
                'hotel1_hotel_161to170 ': hotel1_hotel_161to170,
                'hotel1_hotel_171to180 ': hotel1_hotel_171to180,
                'hotel1_hotel_181to190 ': hotel1_hotel_181to190,
                'hotel1_hotel_191to200 ': hotel1_hotel_191to200,
                'hotel1_hotel_201to210 ': hotel1_hotel_201to210,
                'hotel1_hotel_211to220': hotel1_hotel_211to220,
                'hotel1_hostel': hotel1_hostel,
                'hotel1_hostel_1to10': hotel1_hostel_1to10,
                'hotel1_hostel_11to20 ': hotel1_hostel_11to20,
                'hotel1_hostel_21to30 ': hotel1_hostel_21to30,
                'hotel1_hostel_31to40 ': hotel1_hostel_31to40,
                'hotel1_hostel_41to50 ': hotel1_hostel_41to50,
                'hotel1_hostel_51to60 ': hotel1_hostel_51to60,
                'hotel1_hostel_61to70 ': hotel1_hostel_61to70,
                'hotel1_hostel_71to80 ': hotel1_hostel_71to80,
                'hotel1_hostel_81to90 ': hotel1_hostel_81to90,
                'hotel1_hostel_91to100': hotel1_hostel_91to100,
                'hotel1_hostel_101to110 ': hotel1_hostel_101to110,
                'hotel1_hostel_111to120 ': hotel1_hostel_111to120,
                'hotel1_hostel_121to130 ': hotel1_hostel_121to130,
                'hotel1_hostel_131to140 ': hotel1_hostel_131to140,
                'hotel1_hostel_141to150 ': hotel1_hostel_141to150,
                'hotel1_hostel_151to160 ': hotel1_hostel_151to160,
                'hotel1_hostel_161to170 ': hotel1_hostel_161to170,
                'hotel1_hostel_171to180 ': hotel1_hostel_171to180,
                'hotel1_hostel_181to190 ': hotel1_hostel_181to190,
                'hotel1_hostel_191to200 ': hotel1_hostel_191to200,
                'hotel1_hostel_201to210 ': hotel1_hostel_201to210,
                'hotel1_hostel_211to220': hotel1_hostel_211to220,
                'hotel1_guest': hotel1_guest,
                'hotel1_guest_1to10': hotel1_guest_1to10,
                'hotel1_guest_11to20 ': hotel1_guest_11to20,
                'hotel1_guest_21to30 ': hotel1_guest_21to30,
                'hotel1_guest_31to40 ': hotel1_guest_31to40,
                'hotel1_guest_41to50 ': hotel1_guest_41to50,
                'hotel1_guest_51to60 ': hotel1_guest_51to60,
                'hotel1_guest_61to70 ': hotel1_guest_61to70,
                'hotel1_guest_71to80 ': hotel1_guest_71to80,
                'hotel1_guest_81to90 ': hotel1_guest_81to90,
                'hotel1_guest_91to100': hotel1_guest_91to100,
                'hotel1_guest_101to110 ': hotel1_guest_101to110,
                'hotel1_guest_111to120 ': hotel1_guest_111to120,
                'hotel1_guest_121to130 ': hotel1_guest_121to130,
                'hotel1_guest_131to140 ': hotel1_guest_131to140,
                'hotel1_guest_141to150 ': hotel1_guest_141to150,
                'hotel1_guest_151to160 ': hotel1_guest_151to160,
                'hotel1_guest_161to170 ': hotel1_guest_161to170,
                'hotel1_guest_171to180 ': hotel1_guest_171to180,
                'hotel1_guest_181to190 ': hotel1_guest_181to190,
                'hotel1_guest_191to200 ': hotel1_guest_191to200,
                'hotel1_guest_201to210 ': hotel1_guest_201to210,
                'hotel1_guest_211to220': hotel1_guest_211to220,
                'hotel1_apartment': hotel1_apartment,
                'hotel1_apartment_1to10': hotel1_apartment_1to10,
                'hotel1_apartment_11to20 ': hotel1_apartment_11to20,
                'hotel1_apartment_21to30 ': hotel1_apartment_21to30,
                'hotel1_apartment_31to40 ': hotel1_apartment_31to40,
                'hotel1_apartment_41to50 ': hotel1_apartment_41to50,
                'hotel1_apartment_51to60 ': hotel1_apartment_51to60,
                'hotel1_apartment_61to70 ': hotel1_apartment_61to70,
                'hotel1_apartment_71to80 ': hotel1_apartment_71to80,
                'hotel1_apartment_81to90 ': hotel1_apartment_81to90,
                'hotel1_apartment_91to100': hotel1_apartment_91to100,
                'hotel1_apartment_101to110 ': hotel1_apartment_101to110,
                'hotel1_apartment_111to120 ': hotel1_apartment_111to120,
                'hotel1_apartment_121to130 ': hotel1_apartment_121to130,
                'hotel1_apartment_131to140 ': hotel1_apartment_131to140,
                'hotel1_apartment_141to150 ': hotel1_apartment_141to150,
                'hotel1_apartment_151to160 ': hotel1_apartment_151to160,
                'hotel1_apartment_161to170 ': hotel1_apartment_161to170,
                'hotel1_apartment_171to180 ': hotel1_apartment_171to180,
                'hotel1_apartment_181to190 ': hotel1_apartment_181to190,
                'hotel1_apartment_191to200 ': hotel1_apartment_191to200,
                'hotel1_apartment_201to210 ': hotel1_apartment_201to210,
                'hotel1_apartment_211to220': hotel1_apartment_211to220,
                'hotel1_apartmenthotel': hotel1_apartmenthotel,
                'hotel1_apartmenthotel_1to10': hotel1_apartmenthotel_1to10,
                'hotel1_apartmenthotel_11to20 ': hotel1_apartmenthotel_11to20,
                'hotel1_apartmenthotel_21to30 ': hotel1_apartmenthotel_21to30,
                'hotel1_apartmenthotel_31to40 ': hotel1_apartmenthotel_31to40,
                'hotel1_apartmenthotel_41to50 ': hotel1_apartmenthotel_41to50,
                'hotel1_apartmenthotel_51to60 ': hotel1_apartmenthotel_51to60,
                'hotel1_apartmenthotel_61to70 ': hotel1_apartmenthotel_61to70,
                'hotel1_apartmenthotel_71to80 ': hotel1_apartmenthotel_71to80,
                'hotel1_apartmenthotel_81to90 ': hotel1_apartmenthotel_81to90,
                'hotel1_apartmenthotel_91to100': hotel1_apartmenthotel_91to100,
                'hotel1_apartmenthotel_101to110 ':
                hotel1_apartmenthotel_101to110,
                'hotel1_apartmenthotel_111to120 ':
                hotel1_apartmenthotel_111to120,
                'hotel1_apartmenthotel_121to130 ':
                hotel1_apartmenthotel_121to130,
                'hotel1_apartmenthotel_131to140 ':
                hotel1_apartmenthotel_131to140,
                'hotel1_apartmenthotel_141to150 ':
                hotel1_apartmenthotel_141to150,
                'hotel1_apartmenthotel_151to160 ':
                hotel1_apartmenthotel_151to160,
                'hotel1_apartmenthotel_161to170 ':
                hotel1_apartmenthotel_161to170,
                'hotel1_apartmenthotel_171to180 ':
                hotel1_apartmenthotel_171to180,
                'hotel1_apartmenthotel_181to190 ':
                hotel1_apartmenthotel_181to190,
                'hotel1_apartmenthotel_191to200 ':
                hotel1_apartmenthotel_191to200,
                'hotel1_apartmenthotel_201to210 ':
                hotel1_apartmenthotel_201to210,
                'hotel1_apartmenthotel_211to220':
                hotel1_apartmenthotel_211to220,
                'hotel1_motel': hotel1_motel,
                'hotel1_motel_1to10': hotel1_motel_1to10,
                'hotel1_motel_11to20 ': hotel1_motel_11to20,
                'hotel1_motel_21to30 ': hotel1_motel_21to30,
                'hotel1_motel_31to40 ': hotel1_motel_31to40,
                'hotel1_motel_41to50 ': hotel1_motel_41to50,
                'hotel1_motel_51to60 ': hotel1_motel_51to60,
                'hotel1_motel_61to70 ': hotel1_motel_61to70,
                'hotel1_motel_71to80 ': hotel1_motel_71to80,
                'hotel1_motel_81to90 ': hotel1_motel_81to90,
                'hotel1_motel_91to100': hotel1_motel_91to100,
                'hotel1_motel_101to110 ': hotel1_motel_101to110,
                'hotel1_motel_111to120 ': hotel1_motel_111to120,
                'hotel1_motel_121to130 ': hotel1_motel_121to130,
                'hotel1_motel_131to140 ': hotel1_motel_131to140,
                'hotel1_motel_141to150 ': hotel1_motel_141to150,
                'hotel1_motel_151to160 ': hotel1_motel_151to160,
                'hotel1_motel_161to170 ': hotel1_motel_161to170,
                'hotel1_motel_171to180 ': hotel1_motel_171to180,
                'hotel1_motel_181to190 ': hotel1_motel_181to190,
                'hotel1_motel_191to200 ': hotel1_motel_191to200,
                'hotel1_motel_201to210 ': hotel1_motel_201to210,
                'hotel1_motel_211to220': hotel1_motel_211to220,
                'hotel1_pension': hotel1_pension,
                'hotel1_pension_1to10': hotel1_pension_1to10,
                'hotel1_pension_11to20 ': hotel1_pension_11to20,
                'hotel1_pension_21to30 ': hotel1_pension_21to30,
                'hotel1_pension_31to40 ': hotel1_pension_31to40,
                'hotel1_pension_41to50 ': hotel1_pension_41to50,
                'hotel1_pension_51to60 ': hotel1_pension_51to60,
                'hotel1_pension_61to70 ': hotel1_pension_61to70,
                'hotel1_pension_71to80 ': hotel1_pension_71to80,
                'hotel1_pension_81to90 ': hotel1_pension_81to90,
                'hotel1_pension_91to100': hotel1_pension_91to100,
                'hotel1_pension_101to110 ': hotel1_pension_101to110,
                'hotel1_pension_111to120 ': hotel1_pension_111to120,
                'hotel1_pension_121to130 ': hotel1_pension_121to130,
                'hotel1_pension_131to140 ': hotel1_pension_131to140,
                'hotel1_pension_141to150 ': hotel1_pension_141to150,
                'hotel1_pension_151to160 ': hotel1_pension_151to160,
                'hotel1_pension_161to170 ': hotel1_pension_161to170,
                'hotel1_pension_171to180 ': hotel1_pension_171to180,
                'hotel1_pension_181to190 ': hotel1_pension_181to190,
                'hotel1_pension_191to200 ': hotel1_pension_191to200,
                'hotel1_pension_201to210 ': hotel1_pension_201to210,
                'hotel1_pension_211to220': hotel1_pension_211to220,
                'hotel1_resort': hotel1_resort,
                'hotel1_resort_1to10': hotel1_resort_1to10,
                'hotel1_resort_11to20 ': hotel1_resort_11to20,
                'hotel1_resort_21to30 ': hotel1_resort_21to30,
                'hotel1_resort_31to40 ': hotel1_resort_31to40,
                'hotel1_resort_41to50 ': hotel1_resort_41to50,
                'hotel1_resort_51to60 ': hotel1_resort_51to60,
                'hotel1_resort_61to70 ': hotel1_resort_61to70,
                'hotel1_resort_71to80 ': hotel1_resort_71to80,
                'hotel1_resort_81to90 ': hotel1_resort_81to90,
                'hotel1_resort_91to100': hotel1_resort_91to100,
                'hotel1_resort_101to110 ': hotel1_resort_101to110,
                'hotel1_resort_111to120 ': hotel1_resort_111to120,
                'hotel1_resort_121to130 ': hotel1_resort_121to130,
                'hotel1_resort_131to140 ': hotel1_resort_131to140,
                'hotel1_resort_141to150 ': hotel1_resort_141to150,
                'hotel1_resort_151to160 ': hotel1_resort_151to160,
                'hotel1_resort_161to170 ': hotel1_resort_161to170,
                'hotel1_resort_171to180 ': hotel1_resort_171to180,
                'hotel1_resort_181to190 ': hotel1_resort_181to190,
                'hotel1_resort_191to200 ': hotel1_resort_191to200,
                'hotel1_resort_201to210 ': hotel1_resort_201to210,
                'hotel1_resort_211to220': hotel1_resort_211to220,
                'hotel1_badandbreakfast': hotel1_badandbreakfast,
                'hotel1_badandbreakfast_1to10': hotel1_badandbreakfast_1to10,
                'hotel1_badandbreakfast_11to20 ':
                hotel1_badandbreakfast_11to20,
                'hotel1_badandbreakfast_21to30 ':
                hotel1_badandbreakfast_21to30,
                'hotel1_badandbreakfast_31to40 ':
                hotel1_badandbreakfast_31to40,
                'hotel1_badandbreakfast_41to50 ':
                hotel1_badandbreakfast_41to50,
                'hotel1_badandbreakfast_51to60 ':
                hotel1_badandbreakfast_51to60,
                'hotel1_badandbreakfast_61to70 ':
                hotel1_badandbreakfast_61to70,
                'hotel1_badandbreakfast_71to80 ':
                hotel1_badandbreakfast_71to80,
                'hotel1_badandbreakfast_81to90 ':
                hotel1_badandbreakfast_81to90,
                'hotel1_badandbreakfast_91to100':
                hotel1_badandbreakfast_91to100,
                'hotel1_badandbreakfast_101to110 ':
                hotel1_badandbreakfast_101to110,
                'hotel1_badandbreakfast_111to120 ':
                hotel1_badandbreakfast_111to120,
                'hotel1_badandbreakfast_121to130 ':
                hotel1_badandbreakfast_121to130,
                'hotel1_badandbreakfast_131to140 ':
                hotel1_badandbreakfast_131to140,
                'hotel1_badandbreakfast_141to150 ':
                hotel1_badandbreakfast_141to150,
                'hotel1_badandbreakfast_151to160 ':
                hotel1_badandbreakfast_151to160,
                'hotel1_badandbreakfast_161to170 ':
                hotel1_badandbreakfast_161to170,
                'hotel1_badandbreakfast_171to180 ':
                hotel1_badandbreakfast_171to180,
                'hotel1_badandbreakfast_181to190 ':
                hotel1_badandbreakfast_181to190,
                'hotel1_badandbreakfast_191to200 ':
                hotel1_badandbreakfast_191to200,
                'hotel1_badandbreakfast_201to210 ':
                hotel1_badandbreakfast_201to210,
                'hotel1_badandbreakfast_211to220':
                hotel1_badandbreakfast_211to220,
                'hotel1_homestay': hotel1_homestay,
                'hotel1_homestay_1to10': hotel1_homestay_1to10,
                'hotel1_homestay_11to20 ': hotel1_homestay_11to20,
                'hotel1_homestay_21to30 ': hotel1_homestay_21to30,
                'hotel1_homestay_31to40 ': hotel1_homestay_31to40,
                'hotel1_homestay_41to50 ': hotel1_homestay_41to50,
                'hotel1_homestay_51to60 ': hotel1_homestay_51to60,
                'hotel1_homestay_61to70 ': hotel1_homestay_61to70,
                'hotel1_homestay_71to80 ': hotel1_homestay_71to80,
                'hotel1_homestay_81to90 ': hotel1_homestay_81to90,
                'hotel1_homestay_91to100': hotel1_homestay_91to100,
                'hotel1_homestay_101to110 ': hotel1_homestay_101to110,
                'hotel1_homestay_111to120 ': hotel1_homestay_111to120,
                'hotel1_homestay_121to130 ': hotel1_homestay_121to130,
                'hotel1_homestay_131to140 ': hotel1_homestay_131to140,
                'hotel1_homestay_141to150 ': hotel1_homestay_141to150,
                'hotel1_homestay_151to160 ': hotel1_homestay_151to160,
                'hotel1_homestay_161to170 ': hotel1_homestay_161to170,
                'hotel1_homestay_171to180 ': hotel1_homestay_171to180,
                'hotel1_homestay_181to190 ': hotel1_homestay_181to190,
                'hotel1_homestay_191to200 ': hotel1_homestay_191to200,
                'hotel1_homestay_201to210 ': hotel1_homestay_201to210,
                'hotel1_homestay_211to220': hotel1_homestay_211to220,
                'hotel1_lodge': hotel1_lodge,
                'hotel1_lodge_1to10': hotel1_lodge_1to10,
                'hotel1_lodge_11to20 ': hotel1_lodge_11to20,
                'hotel1_lodge_21to30 ': hotel1_lodge_21to30,
                'hotel1_lodge_31to40 ': hotel1_lodge_31to40,
                'hotel1_lodge_41to50 ': hotel1_lodge_41to50,
                'hotel1_lodge_51to60 ': hotel1_lodge_51to60,
                'hotel1_lodge_61to70 ': hotel1_lodge_61to70,
                'hotel1_lodge_71to80 ': hotel1_lodge_71to80,
                'hotel1_lodge_81to90 ': hotel1_lodge_81to90,
                'hotel1_lodge_91to100': hotel1_lodge_91to100,
                'hotel1_lodge_101to110 ': hotel1_lodge_101to110,
                'hotel1_lodge_111to120 ': hotel1_lodge_111to120,
                'hotel1_lodge_121to130 ': hotel1_lodge_121to130,
                'hotel1_lodge_131to140 ': hotel1_lodge_131to140,
                'hotel1_lodge_141to150 ': hotel1_lodge_141to150,
                'hotel1_lodge_151to160 ': hotel1_lodge_151to160,
                'hotel1_lodge_161to170 ': hotel1_lodge_161to170,
                'hotel1_lodge_171to180 ': hotel1_lodge_171to180,
                'hotel1_lodge_181to190 ': hotel1_lodge_181to190,
                'hotel1_lodge_191to200 ': hotel1_lodge_191to200,
                'hotel1_lodge_201to210 ': hotel1_lodge_201to210,
                'hotel1_lodge_211to220': hotel1_lodge_211to220,
                'hotel1_countryhouse': hotel1_countryhouse,
                'hotel1_countryhouse_1to10': hotel1_countryhouse_1to10,
                'hotel1_countryhouse_11to20 ': hotel1_countryhouse_11to20,
                'hotel1_countryhouse_21to30 ': hotel1_countryhouse_21to30,
                'hotel1_countryhouse_31to40 ': hotel1_countryhouse_31to40,
                'hotel1_countryhouse_41to50 ': hotel1_countryhouse_41to50,
                'hotel1_countryhouse_51to60 ': hotel1_countryhouse_51to60,
                'hotel1_countryhouse_61to70 ': hotel1_countryhouse_61to70,
                'hotel1_countryhouse_71to80 ': hotel1_countryhouse_71to80,
                'hotel1_countryhouse_81to90 ': hotel1_countryhouse_81to90,
                'hotel1_countryhouse_91to100': hotel1_countryhouse_91to100,
                'hotel1_countryhouse_101to110 ': hotel1_countryhouse_101to110,
                'hotel1_countryhouse_111to120 ': hotel1_countryhouse_111to120,
                'hotel1_countryhouse_121to130 ': hotel1_countryhouse_121to130,
                'hotel1_countryhouse_131to140 ': hotel1_countryhouse_131to140,
                'hotel1_countryhouse_141to150 ': hotel1_countryhouse_141to150,
                'hotel1_countryhouse_151to160 ': hotel1_countryhouse_151to160,
                'hotel1_countryhouse_161to170 ': hotel1_countryhouse_161to170,
                'hotel1_countryhouse_171to180 ': hotel1_countryhouse_171to180,
                'hotel1_countryhouse_181to190 ': hotel1_countryhouse_181to190,
                'hotel1_countryhouse_191to200 ': hotel1_countryhouse_191to200,
                'hotel1_countryhouse_201to210 ': hotel1_countryhouse_201to210,
                'hotel1_countryhouse_211to220': hotel1_countryhouse_211to220,
                'hotel1_inn': hotel1_inn,
                'hotel1_inn_1to10': hotel1_inn_1to10,
                'hotel1_inn_11to20 ': hotel1_inn_11to20,
                'hotel1_inn_21to30 ': hotel1_inn_21to30,
                'hotel1_inn_31to40 ': hotel1_inn_31to40,
                'hotel1_inn_41to50 ': hotel1_inn_41to50,
                'hotel1_inn_51to60 ': hotel1_inn_51to60,
                'hotel1_inn_61to70 ': hotel1_inn_61to70,
                'hotel1_inn_71to80 ': hotel1_inn_71to80,
                'hotel1_inn_81to90 ': hotel1_inn_81to90,
                'hotel1_inn_91to100': hotel1_inn_91to100,
                'hotel1_inn_101to110 ': hotel1_inn_101to110,
                'hotel1_inn_111to120 ': hotel1_inn_111to120,
                'hotel1_inn_121to130 ': hotel1_inn_121to130,
                'hotel1_inn_131to140 ': hotel1_inn_131to140,
                'hotel1_inn_141to150 ': hotel1_inn_141to150,
                'hotel1_inn_151to160 ': hotel1_inn_151to160,
                'hotel1_inn_161to170 ': hotel1_inn_161to170,
                'hotel1_inn_171to180 ': hotel1_inn_171to180,
                'hotel1_inn_181to190 ': hotel1_inn_181to190,
                'hotel1_inn_191to200 ': hotel1_inn_191to200,
                'hotel1_inn_201to210 ': hotel1_inn_201to210,
                'hotel1_inn_211to220': hotel1_inn_211to220,
                'hotel1_villa': hotel1_villa,
                'hotel1_villa_1to10': hotel1_villa_1to10,
                'hotel1_villa_11to20 ': hotel1_villa_11to20,
                'hotel1_villa_21to30 ': hotel1_villa_21to30,
                'hotel1_villa_31to40 ': hotel1_villa_31to40,
                'hotel1_villa_41to50 ': hotel1_villa_41to50,
                'hotel1_villa_51to60 ': hotel1_villa_51to60,
                'hotel1_villa_61to70 ': hotel1_villa_61to70,
                'hotel1_villa_71to80 ': hotel1_villa_71to80,
                'hotel1_villa_81to90 ': hotel1_villa_81to90,
                'hotel1_villa_91to100': hotel1_villa_91to100,
                'hotel1_villa_101to110 ': hotel1_villa_101to110,
                'hotel1_villa_111to120 ': hotel1_villa_111to120,
                'hotel1_villa_121to130 ': hotel1_villa_121to130,
                'hotel1_villa_131to140 ': hotel1_villa_131to140,
                'hotel1_villa_141to150 ': hotel1_villa_141to150,
                'hotel1_villa_151to160 ': hotel1_villa_151to160,
                'hotel1_villa_161to170 ': hotel1_villa_161to170,
                'hotel1_villa_171to180 ': hotel1_villa_171to180,
                'hotel1_villa_181to190 ': hotel1_villa_181to190,
                'hotel1_villa_191to200 ': hotel1_villa_191to200,
                'hotel1_villa_201to210 ': hotel1_villa_201to210,
                'hotel1_villa_211to220': hotel1_villa_211to220,
                'hotel1_camping': hotel1_camping,
                'hotel1_camping_1to10': hotel1_camping_1to10,
                'hotel1_camping_11to20 ': hotel1_camping_11to20,
                'hotel1_camping_21to30 ': hotel1_camping_21to30,
                'hotel1_camping_31to40 ': hotel1_camping_31to40,
                'hotel1_camping_41to50 ': hotel1_camping_41to50,
                'hotel1_camping_51to60 ': hotel1_camping_51to60,
                'hotel1_camping_61to70 ': hotel1_camping_61to70,
                'hotel1_camping_71to80 ': hotel1_camping_71to80,
                'hotel1_camping_81to90 ': hotel1_camping_81to90,
                'hotel1_camping_91to100': hotel1_camping_91to100,
                'hotel1_camping_101to110 ': hotel1_camping_101to110,
                'hotel1_camping_111to120 ': hotel1_camping_111to120,
                'hotel1_camping_121to130 ': hotel1_camping_121to130,
                'hotel1_camping_131to140 ': hotel1_camping_131to140,
                'hotel1_camping_141to150 ': hotel1_camping_141to150,
                'hotel1_camping_151to160 ': hotel1_camping_151to160,
                'hotel1_camping_161to170 ': hotel1_camping_161to170,
                'hotel1_camping_171to180 ': hotel1_camping_171to180,
                'hotel1_camping_181to190 ': hotel1_camping_181to190,
                'hotel1_camping_191to200 ': hotel1_camping_191to200,
                'hotel1_camping_201to210 ': hotel1_camping_201to210,
                'hotel1_camping_211to220': hotel1_camping_211to220,
                'hotels2': hotel2,
                'hotel2_1to10 ': hotel2_1to10,
                'hotel2_11to20': hotel2_11to20,
                'hotel2_21to30': hotel2_21to30,
                'hotel2_31to40': hotel2_31to40,
                'hotel2_41to50': hotel2_41to50,
                'hotel2_51to60': hotel2_51to60,
                'hotel2_61to70': hotel2_61to70,
                'hotel2_71to80': hotel2_71to80,
                'hotel2_81to90': hotel2_81to90,
                'hotel2_91to100': hotel2_91to100,
                'hotel2_101to110': hotel2_101to110,
                'hotel2_111to120': hotel2_111to120,
                'hotel2_121to130': hotel2_121to130,
                'hotel2_131to140': hotel2_131to140,
                'hotel2_141to150': hotel2_141to150,
                'hotel2_151to160': hotel2_151to160,
                'hotel2_161to170': hotel2_161to170,
                'hotel2_171to180': hotel2_171to180,
                'hotel2_181to190': hotel2_181to190,
                'hotel2_191to200': hotel2_191to200,
                'hotel2_201to210': hotel2_201to210,
                'hotel2_211to220': hotel2_211to220,
                'hotel2_cost_up': hotel2_cost_up,
                'hotel2_cost_up_1to10': hotel2_cost_up_1to10,
                'hotel2_cost_up_11to20 ': hotel2_cost_up_11to20,
                'hotel2_cost_up_21to30 ': hotel2_cost_up_21to30,
                'hotel2_cost_up_31to40 ': hotel2_cost_up_31to40,
                'hotel2_cost_up_41to50 ': hotel2_cost_up_41to50,
                'hotel2_cost_up_51to60 ': hotel2_cost_up_51to60,
                'hotel2_cost_up_61to70 ': hotel2_cost_up_61to70,
                'hotel2_cost_up_71to80 ': hotel2_cost_up_71to80,
                'hotel2_cost_up_81to90 ': hotel2_cost_up_81to90,
                'hotel2_cost_up_91to100': hotel2_cost_up_91to100,
                'hotel2_cost_up_101to110': hotel2_cost_up_101to110,
                'hotel2_cost_up_111to120': hotel2_cost_up_111to120,
                'hotel2_cost_up_121to130': hotel2_cost_up_121to130,
                'hotel2_cost_up_131to140': hotel2_cost_up_131to140,
                'hotel2_cost_up_141to150': hotel2_cost_up_141to150,
                'hotel2_cost_up_151to160': hotel2_cost_up_151to160,
                'hotel2_cost_up_161to170': hotel2_cost_up_161to170,
                'hotel2_cost_up_171to180': hotel2_cost_up_171to180,
                'hotel2_cost_up_181to190': hotel2_cost_up_181to190,
                'hotel2_cost_up_191to200': hotel2_cost_up_191to200,
                'hotel2_cost_up_201to210': hotel2_cost_up_201to210,
                'hotel2_cost_up_211to220': hotel2_cost_up_211to220,
                'hotel2_cost_down': hotel2_cost_down,
                'hotel2_cost_down_1to10': hotel2_cost_down_1to10,
                'hotel2_cost_down_11to20 ': hotel2_cost_down_11to20,
                'hotel2_cost_down_21to30 ': hotel2_cost_down_21to30,
                'hotel2_cost_down_31to40 ': hotel2_cost_down_31to40,
                'hotel2_cost_down_41to50 ': hotel2_cost_down_41to50,
                'hotel2_cost_down_51to60 ': hotel2_cost_down_51to60,
                'hotel2_cost_down_61to70 ': hotel2_cost_down_61to70,
                'hotel2_cost_down_71to80 ': hotel2_cost_down_71to80,
                'hotel2_cost_down_81to90 ': hotel2_cost_down_81to90,
                'hotel2_cost_down_91to100': hotel2_cost_down_91to100,
                'hotel2_cost_down_101to110': hotel2_cost_down_101to110,
                'hotel2_cost_down_111to120': hotel2_cost_down_111to120,
                'hotel2_cost_down_121to130': hotel2_cost_down_121to130,
                'hotel2_cost_down_131to140': hotel2_cost_down_131to140,
                'hotel2_cost_down_141to150': hotel2_cost_down_141to150,
                'hotel2_cost_down_151to160': hotel2_cost_down_151to160,
                'hotel2_cost_down_161to170': hotel2_cost_down_161to170,
                'hotel2_cost_down_171to180': hotel2_cost_down_171to180,
                'hotel2_cost_down_181to190': hotel2_cost_down_181to190,
                'hotel2_cost_down_191to200': hotel2_cost_down_191to200,
                'hotel2_cost_down_201to210': hotel2_cost_down_201to210,
                'hotel2_cost_down_211to220': hotel2_cost_down_211to220,
                'hotel2_rating_down': hotel2_rating_down,
                'hotel2_rating_down_1to10': hotel2_rating_down_1to10,
                'hotel2_rating_down_11to20 ': hotel2_rating_down_11to20,
                'hotel2_rating_down_21to30 ': hotel2_rating_down_21to30,
                'hotel2_rating_down_31to40 ': hotel2_rating_down_31to40,
                'hotel2_rating_down_41to50 ': hotel2_rating_down_41to50,
                'hotel2_rating_down_51to60 ': hotel2_rating_down_51to60,
                'hotel2_rating_down_61to70 ': hotel2_rating_down_61to70,
                'hotel2_rating_down_71to80 ': hotel2_rating_down_71to80,
                'hotel2_rating_down_81to90 ': hotel2_rating_down_81to90,
                'hotel2_rating_down_91to100': hotel2_rating_down_91to100,
                'hotel2_rating_down_101to110': hotel2_rating_down_101to110,
                'hotel2_rating_down_111to120': hotel2_rating_down_111to120,
                'hotel2_rating_down_121to130': hotel2_rating_down_121to130,
                'hotel2_rating_down_131to140': hotel2_rating_down_131to140,
                'hotel2_rating_down_141to150': hotel2_rating_down_141to150,
                'hotel2_rating_down_151to160': hotel2_rating_down_151to160,
                'hotel2_rating_down_161to170': hotel2_rating_down_161to170,
                'hotel2_rating_down_171to180': hotel2_rating_down_171to180,
                'hotel2_rating_down_181to190': hotel2_rating_down_181to190,
                'hotel2_rating_down_191to200': hotel2_rating_down_191to200,
                'hotel2_rating_down_201to210': hotel2_rating_down_201to210,
                'hotel2_rating_down_211to220': hotel2_rating_down_211to220,
                'hotel2_distance_up': hotel2_distance_up,
                'hotel2_distance_up_1to10': hotel2_distance_up_1to10,
                'hotel2_distance_up_11to20 ': hotel2_distance_up_11to20,
                'hotel2_distance_up_21to30 ': hotel2_distance_up_21to30,
                'hotel2_distance_up_31to40 ': hotel2_distance_up_31to40,
                'hotel2_distance_up_41to50 ': hotel2_distance_up_41to50,
                'hotel2_distance_up_51to60 ': hotel2_distance_up_51to60,
                'hotel2_distance_up_61to70 ': hotel2_distance_up_61to70,
                'hotel2_distance_up_71to80 ': hotel2_distance_up_71to80,
                'hotel2_distance_up_81to90 ': hotel2_distance_up_81to90,
                'hotel2_distance_up_91to100': hotel2_distance_up_91to100,
                'hotel2_distance_up_101to110 ': hotel2_distance_up_101to110,
                'hotel2_distance_up_111to120 ': hotel2_distance_up_111to120,
                'hotel2_distance_up_121to130 ': hotel2_distance_up_121to130,
                'hotel2_distance_up_131to140 ': hotel2_distance_up_131to140,
                'hotel2_distance_up_141to150 ': hotel2_distance_up_141to150,
                'hotel2_distance_up_151to160 ': hotel2_distance_up_151to160,
                'hotel2_distance_up_161to170 ': hotel2_distance_up_161to170,
                'hotel2_distance_up_171to180 ': hotel2_distance_up_171to180,
                'hotel2_distance_up_181to190 ': hotel2_distance_up_181to190,
                'hotel2_distance_up_191to200 ': hotel2_distance_up_191to200,
                'hotel2_distance_up_201to210 ': hotel2_distance_up_201to210,
                'hotel2_distance_up_211to220': hotel2_distance_up_211to220,
                'hotel2_kind_down_1to10': hotel2_kind_down_1to10,
                'hotel2_kind_down_11to20 ': hotel2_kind_down_11to20,
                'hotel2_kind_down_21to30 ': hotel2_kind_down_21to30,
                'hotel2_kind_down_31to40 ': hotel2_kind_down_31to40,
                'hotel2_kind_down_41to50 ': hotel2_kind_down_41to50,
                'hotel2_kind_down_51to60 ': hotel2_kind_down_51to60,
                'hotel2_kind_down_61to70 ': hotel2_kind_down_61to70,
                'hotel2_kind_down_71to80 ': hotel2_kind_down_71to80,
                'hotel2_kind_down_81to90 ': hotel2_kind_down_81to90,
                'hotel2_kind_down_91to100': hotel2_kind_down_91to100,
                'hotel2_kind_down_101to110 ': hotel2_kind_down_101to110,
                'hotel2_kind_down_111to120 ': hotel2_kind_down_111to120,
                'hotel2_kind_down_121to130 ': hotel2_kind_down_121to130,
                'hotel2_kind_down_131to140 ': hotel2_kind_down_131to140,
                'hotel2_kind_down_141to150 ': hotel2_kind_down_141to150,
                'hotel2_kind_down_151to160 ': hotel2_kind_down_151to160,
                'hotel2_kind_down_161to170 ': hotel2_kind_down_161to170,
                'hotel2_kind_down_171to180 ': hotel2_kind_down_171to180,
                'hotel2_kind_down_181to190 ': hotel2_kind_down_181to190,
                'hotel2_kind_down_191to200 ': hotel2_kind_down_191to200,
                'hotel2_kind_down_201to210 ': hotel2_kind_down_201to210,
                'hotel2_kind_down_211to220': hotel2_kind_down_211to220,
                'hotel2_clean_down_1to10': hotel2_clean_down_1to10,
                'hotel2_clean_down_11to20 ': hotel2_clean_down_11to20,
                'hotel2_clean_down_21to30 ': hotel2_clean_down_21to30,
                'hotel2_clean_down_31to40 ': hotel2_clean_down_31to40,
                'hotel2_clean_down_41to50 ': hotel2_clean_down_41to50,
                'hotel2_clean_down_51to60 ': hotel2_clean_down_51to60,
                'hotel2_clean_down_61to70 ': hotel2_clean_down_61to70,
                'hotel2_clean_down_71to80 ': hotel2_clean_down_71to80,
                'hotel2_clean_down_81to90 ': hotel2_clean_down_81to90,
                'hotel2_clean_down_91to100': hotel2_clean_down_91to100,
                'hotel2_clean_down_101to110 ': hotel2_clean_down_101to110,
                'hotel2_clean_down_111to120 ': hotel2_clean_down_111to120,
                'hotel2_clean_down_121to130 ': hotel2_clean_down_121to130,
                'hotel2_clean_down_131to140 ': hotel2_clean_down_131to140,
                'hotel2_clean_down_141to150 ': hotel2_clean_down_141to150,
                'hotel2_clean_down_151to160 ': hotel2_clean_down_151to160,
                'hotel2_clean_down_161to170 ': hotel2_clean_down_161to170,
                'hotel2_clean_down_171to180 ': hotel2_clean_down_171to180,
                'hotel2_clean_down_181to190 ': hotel2_clean_down_181to190,
                'hotel2_clean_down_191to200 ': hotel2_clean_down_191to200,
                'hotel2_clean_down_201to210 ': hotel2_clean_down_201to210,
                'hotel2_clean_down_211to220': hotel2_clean_down_211to220,
                'hotel2_conv_down_1to10': hotel2_conv_down_1to10,
                'hotel2_conv_down_11to20 ': hotel2_conv_down_11to20,
                'hotel2_conv_down_21to30 ': hotel2_conv_down_21to30,
                'hotel2_conv_down_31to40 ': hotel2_conv_down_31to40,
                'hotel2_conv_down_41to50 ': hotel2_conv_down_41to50,
                'hotel2_conv_down_51to60 ': hotel2_conv_down_51to60,
                'hotel2_conv_down_61to70 ': hotel2_conv_down_61to70,
                'hotel2_conv_down_71to80 ': hotel2_conv_down_71to80,
                'hotel2_conv_down_81to90 ': hotel2_conv_down_81to90,
                'hotel2_conv_down_91to100': hotel2_conv_down_91to100,
                'hotel2_conv_down_101to110 ': hotel2_conv_down_101to110,
                'hotel2_conv_down_111to120 ': hotel2_conv_down_111to120,
                'hotel2_conv_down_121to130 ': hotel2_conv_down_121to130,
                'hotel2_conv_down_131to140 ': hotel2_conv_down_131to140,
                'hotel2_conv_down_141to150 ': hotel2_conv_down_141to150,
                'hotel2_conv_down_151to160 ': hotel2_conv_down_151to160,
                'hotel2_conv_down_161to170 ': hotel2_conv_down_161to170,
                'hotel2_conv_down_171to180 ': hotel2_conv_down_171to180,
                'hotel2_conv_down_181to190 ': hotel2_conv_down_181to190,
                'hotel2_conv_down_191to200 ': hotel2_conv_down_191to200,
                'hotel2_conv_down_201to210 ': hotel2_conv_down_201to210,
                'hotel2_conv_down_211to220': hotel2_conv_down_211to220,
                'hotel2_hotel': hotel2_hotel,
                'hotel2_hotel_1to10': hotel2_hotel_1to10,
                'hotel2_hotel_11to20 ': hotel2_hotel_11to20,
                'hotel2_hotel_21to30 ': hotel2_hotel_21to30,
                'hotel2_hotel_31to40 ': hotel2_hotel_31to40,
                'hotel2_hotel_41to50 ': hotel2_hotel_41to50,
                'hotel2_hotel_51to60 ': hotel2_hotel_51to60,
                'hotel2_hotel_61to70 ': hotel2_hotel_61to70,
                'hotel2_hotel_71to80 ': hotel2_hotel_71to80,
                'hotel2_hotel_81to90 ': hotel2_hotel_81to90,
                'hotel2_hotel_91to100': hotel2_hotel_91to100,
                'hotel2_hotel_101to110 ': hotel2_hotel_101to110,
                'hotel2_hotel_111to120 ': hotel2_hotel_111to120,
                'hotel2_hotel_121to130 ': hotel2_hotel_121to130,
                'hotel2_hotel_131to140 ': hotel2_hotel_131to140,
                'hotel2_hotel_141to150 ': hotel2_hotel_141to150,
                'hotel2_hotel_151to160 ': hotel2_hotel_151to160,
                'hotel2_hotel_161to170 ': hotel2_hotel_161to170,
                'hotel2_hotel_171to180 ': hotel2_hotel_171to180,
                'hotel2_hotel_181to190 ': hotel2_hotel_181to190,
                'hotel2_hotel_191to200 ': hotel2_hotel_191to200,
                'hotel2_hotel_201to210 ': hotel2_hotel_201to210,
                'hotel2_hotel_211to220': hotel2_hotel_211to220,
                'hotel2_hostel': hotel2_hostel,
                'hotel2_hostel_1to10': hotel2_hostel_1to10,
                'hotel2_hostel_11to20 ': hotel2_hostel_11to20,
                'hotel2_hostel_21to30 ': hotel2_hostel_21to30,
                'hotel2_hostel_31to40 ': hotel2_hostel_31to40,
                'hotel2_hostel_41to50 ': hotel2_hostel_41to50,
                'hotel2_hostel_51to60 ': hotel2_hostel_51to60,
                'hotel2_hostel_61to70 ': hotel2_hostel_61to70,
                'hotel2_hostel_71to80 ': hotel2_hostel_71to80,
                'hotel2_hostel_81to90 ': hotel2_hostel_81to90,
                'hotel2_hostel_91to100': hotel2_hostel_91to100,
                'hotel2_hostel_101to110 ': hotel2_hostel_101to110,
                'hotel2_hostel_111to120 ': hotel2_hostel_111to120,
                'hotel2_hostel_121to130 ': hotel2_hostel_121to130,
                'hotel2_hostel_131to140 ': hotel2_hostel_131to140,
                'hotel2_hostel_141to150 ': hotel2_hostel_141to150,
                'hotel2_hostel_151to160 ': hotel2_hostel_151to160,
                'hotel2_hostel_161to170 ': hotel2_hostel_161to170,
                'hotel2_hostel_171to180 ': hotel2_hostel_171to180,
                'hotel2_hostel_181to190 ': hotel2_hostel_181to190,
                'hotel2_hostel_191to200 ': hotel2_hostel_191to200,
                'hotel2_hostel_201to210 ': hotel2_hostel_201to210,
                'hotel2_hostel_211to220': hotel2_hostel_211to220,
                'hotel2_guest': hotel2_guest,
                'hotel2_guest_1to10': hotel2_guest_1to10,
                'hotel2_guest_11to20 ': hotel2_guest_11to20,
                'hotel2_guest_21to30 ': hotel2_guest_21to30,
                'hotel2_guest_31to40 ': hotel2_guest_31to40,
                'hotel2_guest_41to50 ': hotel2_guest_41to50,
                'hotel2_guest_51to60 ': hotel2_guest_51to60,
                'hotel2_guest_61to70 ': hotel2_guest_61to70,
                'hotel2_guest_71to80 ': hotel2_guest_71to80,
                'hotel2_guest_81to90 ': hotel2_guest_81to90,
                'hotel2_guest_91to100': hotel2_guest_91to100,
                'hotel2_guest_101to110 ': hotel2_guest_101to110,
                'hotel2_guest_111to120 ': hotel2_guest_111to120,
                'hotel2_guest_121to130 ': hotel2_guest_121to130,
                'hotel2_guest_131to140 ': hotel2_guest_131to140,
                'hotel2_guest_141to150 ': hotel2_guest_141to150,
                'hotel2_guest_151to160 ': hotel2_guest_151to160,
                'hotel2_guest_161to170 ': hotel2_guest_161to170,
                'hotel2_guest_171to180 ': hotel2_guest_171to180,
                'hotel2_guest_181to190 ': hotel2_guest_181to190,
                'hotel2_guest_191to200 ': hotel2_guest_191to200,
                'hotel2_guest_201to210 ': hotel2_guest_201to210,
                'hotel2_guest_211to220': hotel2_guest_211to220,
                'hotel2_apartment': hotel2_apartment,
                'hotel2_apartment_1to10': hotel2_apartment_1to10,
                'hotel2_apartment_11to20 ': hotel2_apartment_11to20,
                'hotel2_apartment_21to30 ': hotel2_apartment_21to30,
                'hotel2_apartment_31to40 ': hotel2_apartment_31to40,
                'hotel2_apartment_41to50 ': hotel2_apartment_41to50,
                'hotel2_apartment_51to60 ': hotel2_apartment_51to60,
                'hotel2_apartment_61to70 ': hotel2_apartment_61to70,
                'hotel2_apartment_71to80 ': hotel2_apartment_71to80,
                'hotel2_apartment_81to90 ': hotel2_apartment_81to90,
                'hotel2_apartment_91to100': hotel2_apartment_91to100,
                'hotel2_apartment_101to110 ': hotel2_apartment_101to110,
                'hotel2_apartment_111to120 ': hotel2_apartment_111to120,
                'hotel2_apartment_121to130 ': hotel2_apartment_121to130,
                'hotel2_apartment_131to140 ': hotel2_apartment_131to140,
                'hotel2_apartment_141to150 ': hotel2_apartment_141to150,
                'hotel2_apartment_151to160 ': hotel2_apartment_151to160,
                'hotel2_apartment_161to170 ': hotel2_apartment_161to170,
                'hotel2_apartment_171to180 ': hotel2_apartment_171to180,
                'hotel2_apartment_181to190 ': hotel2_apartment_181to190,
                'hotel2_apartment_191to200 ': hotel2_apartment_191to200,
                'hotel2_apartment_201to210 ': hotel2_apartment_201to210,
                'hotel2_apartment_211to220': hotel2_apartment_211to220,
                'hotel2_apartmenthotel': hotel2_apartmenthotel,
                'hotel2_apartmenthotel_1to10': hotel2_apartmenthotel_1to10,
                'hotel2_apartmenthotel_11to20 ': hotel2_apartmenthotel_11to20,
                'hotel2_apartmenthotel_21to30 ': hotel2_apartmenthotel_21to30,
                'hotel2_apartmenthotel_31to40 ': hotel2_apartmenthotel_31to40,
                'hotel2_apartmenthotel_41to50 ': hotel2_apartmenthotel_41to50,
                'hotel2_apartmenthotel_51to60 ': hotel2_apartmenthotel_51to60,
                'hotel2_apartmenthotel_61to70 ': hotel2_apartmenthotel_61to70,
                'hotel2_apartmenthotel_71to80 ': hotel2_apartmenthotel_71to80,
                'hotel2_apartmenthotel_81to90 ': hotel2_apartmenthotel_81to90,
                'hotel2_apartmenthotel_91to100': hotel2_apartmenthotel_91to100,
                'hotel2_apartmenthotel_101to110 ':
                hotel2_apartmenthotel_101to110,
                'hotel2_apartmenthotel_111to120 ':
                hotel2_apartmenthotel_111to120,
                'hotel2_apartmenthotel_121to130 ':
                hotel2_apartmenthotel_121to130,
                'hotel2_apartmenthotel_131to140 ':
                hotel2_apartmenthotel_131to140,
                'hotel2_apartmenthotel_141to150 ':
                hotel2_apartmenthotel_141to150,
                'hotel2_apartmenthotel_151to160 ':
                hotel2_apartmenthotel_151to160,
                'hotel2_apartmenthotel_161to170 ':
                hotel2_apartmenthotel_161to170,
                'hotel2_apartmenthotel_171to180 ':
                hotel2_apartmenthotel_171to180,
                'hotel2_apartmenthotel_181to190 ':
                hotel2_apartmenthotel_181to190,
                'hotel2_apartmenthotel_191to200 ':
                hotel2_apartmenthotel_191to200,
                'hotel2_apartmenthotel_201to210 ':
                hotel2_apartmenthotel_201to210,
                'hotel2_apartmenthotel_211to220':
                hotel2_apartmenthotel_211to220,
                'hotel2_motel': hotel2_motel,
                'hotel2_motel_1to10': hotel2_motel_1to10,
                'hotel2_motel_11to20 ': hotel2_motel_11to20,
                'hotel2_motel_21to30 ': hotel2_motel_21to30,
                'hotel2_motel_31to40 ': hotel2_motel_31to40,
                'hotel2_motel_41to50 ': hotel2_motel_41to50,
                'hotel2_motel_51to60 ': hotel2_motel_51to60,
                'hotel2_motel_61to70 ': hotel2_motel_61to70,
                'hotel2_motel_71to80 ': hotel2_motel_71to80,
                'hotel2_motel_81to90 ': hotel2_motel_81to90,
                'hotel2_motel_91to100': hotel2_motel_91to100,
                'hotel2_motel_101to110 ': hotel2_motel_101to110,
                'hotel2_motel_111to120 ': hotel2_motel_111to120,
                'hotel2_motel_121to130 ': hotel2_motel_121to130,
                'hotel2_motel_131to140 ': hotel2_motel_131to140,
                'hotel2_motel_141to150 ': hotel2_motel_141to150,
                'hotel2_motel_151to160 ': hotel2_motel_151to160,
                'hotel2_motel_161to170 ': hotel2_motel_161to170,
                'hotel2_motel_171to180 ': hotel2_motel_171to180,
                'hotel2_motel_181to190 ': hotel2_motel_181to190,
                'hotel2_motel_191to200 ': hotel2_motel_191to200,
                'hotel2_motel_201to210 ': hotel2_motel_201to210,
                'hotel2_motel_211to220': hotel2_motel_211to220,
                'hotel2_pension': hotel2_pension,
                'hotel2_pension_1to10': hotel2_pension_1to10,
                'hotel2_pension_11to20 ': hotel2_pension_11to20,
                'hotel2_pension_21to30 ': hotel2_pension_21to30,
                'hotel2_pension_31to40 ': hotel2_pension_31to40,
                'hotel2_pension_41to50 ': hotel2_pension_41to50,
                'hotel2_pension_51to60 ': hotel2_pension_51to60,
                'hotel2_pension_61to70 ': hotel2_pension_61to70,
                'hotel2_pension_71to80 ': hotel2_pension_71to80,
                'hotel2_pension_81to90 ': hotel2_pension_81to90,
                'hotel2_pension_91to100': hotel2_pension_91to100,
                'hotel2_pension_101to110 ': hotel2_pension_101to110,
                'hotel2_pension_111to120 ': hotel2_pension_111to120,
                'hotel2_pension_121to130 ': hotel2_pension_121to130,
                'hotel2_pension_131to140 ': hotel2_pension_131to140,
                'hotel2_pension_141to150 ': hotel2_pension_141to150,
                'hotel2_pension_151to160 ': hotel2_pension_151to160,
                'hotel2_pension_161to170 ': hotel2_pension_161to170,
                'hotel2_pension_171to180 ': hotel2_pension_171to180,
                'hotel2_pension_181to190 ': hotel2_pension_181to190,
                'hotel2_pension_191to200 ': hotel2_pension_191to200,
                'hotel2_pension_201to210 ': hotel2_pension_201to210,
                'hotel2_pension_211to220': hotel2_pension_211to220,
                'hotel2_resort': hotel2_resort,
                'hotel2_resort_1to10': hotel2_resort_1to10,
                'hotel2_resort_11to20 ': hotel2_resort_11to20,
                'hotel2_resort_21to30 ': hotel2_resort_21to30,
                'hotel2_resort_31to40 ': hotel2_resort_31to40,
                'hotel2_resort_41to50 ': hotel2_resort_41to50,
                'hotel2_resort_51to60 ': hotel2_resort_51to60,
                'hotel2_resort_61to70 ': hotel2_resort_61to70,
                'hotel2_resort_71to80 ': hotel2_resort_71to80,
                'hotel2_resort_81to90 ': hotel2_resort_81to90,
                'hotel2_resort_91to100': hotel2_resort_91to100,
                'hotel2_resort_101to110 ': hotel2_resort_101to110,
                'hotel2_resort_111to120 ': hotel2_resort_111to120,
                'hotel2_resort_121to130 ': hotel2_resort_121to130,
                'hotel2_resort_131to140 ': hotel2_resort_131to140,
                'hotel2_resort_141to150 ': hotel2_resort_141to150,
                'hotel2_resort_151to160 ': hotel2_resort_151to160,
                'hotel2_resort_161to170 ': hotel2_resort_161to170,
                'hotel2_resort_171to180 ': hotel2_resort_171to180,
                'hotel2_resort_181to190 ': hotel2_resort_181to190,
                'hotel2_resort_191to200 ': hotel2_resort_191to200,
                'hotel2_resort_201to210 ': hotel2_resort_201to210,
                'hotel2_resort_211to220': hotel2_resort_211to220,
                'hotel2_badandbreakfast': hotel2_badandbreakfast,
                'hotel2_badandbreakfast_1to10': hotel2_badandbreakfast_1to10,
                'hotel2_badandbreakfast_11to20 ':
                hotel2_badandbreakfast_11to20,
                'hotel2_badandbreakfast_21to30 ':
                hotel2_badandbreakfast_21to30,
                'hotel2_badandbreakfast_31to40 ':
                hotel2_badandbreakfast_31to40,
                'hotel2_badandbreakfast_41to50 ':
                hotel2_badandbreakfast_41to50,
                'hotel2_badandbreakfast_51to60 ':
                hotel2_badandbreakfast_51to60,
                'hotel2_badandbreakfast_61to70 ':
                hotel2_badandbreakfast_61to70,
                'hotel2_badandbreakfast_71to80 ':
                hotel2_badandbreakfast_71to80,
                'hotel2_badandbreakfast_81to90 ':
                hotel2_badandbreakfast_81to90,
                'hotel2_badandbreakfast_91to100':
                hotel2_badandbreakfast_91to100,
                'hotel2_badandbreakfast_101to110 ':
                hotel2_badandbreakfast_101to110,
                'hotel2_badandbreakfast_111to120 ':
                hotel2_badandbreakfast_111to120,
                'hotel2_badandbreakfast_121to130 ':
                hotel2_badandbreakfast_121to130,
                'hotel2_badandbreakfast_131to140 ':
                hotel2_badandbreakfast_131to140,
                'hotel2_badandbreakfast_141to150 ':
                hotel2_badandbreakfast_141to150,
                'hotel2_badandbreakfast_151to160 ':
                hotel2_badandbreakfast_151to160,
                'hotel2_badandbreakfast_161to170 ':
                hotel2_badandbreakfast_161to170,
                'hotel2_badandbreakfast_171to180 ':
                hotel2_badandbreakfast_171to180,
                'hotel2_badandbreakfast_181to190 ':
                hotel2_badandbreakfast_181to190,
                'hotel2_badandbreakfast_191to200 ':
                hotel2_badandbreakfast_191to200,
                'hotel2_badandbreakfast_201to210 ':
                hotel2_badandbreakfast_201to210,
                'hotel2_badandbreakfast_211to220':
                hotel2_badandbreakfast_211to220,
                'hotel2_homestay': hotel2_homestay,
                'hotel2_homestay_1to10': hotel2_homestay_1to10,
                'hotel2_homestay_11to20 ': hotel2_homestay_11to20,
                'hotel2_homestay_21to30 ': hotel2_homestay_21to30,
                'hotel2_homestay_31to40 ': hotel2_homestay_31to40,
                'hotel2_homestay_41to50 ': hotel2_homestay_41to50,
                'hotel2_homestay_51to60 ': hotel2_homestay_51to60,
                'hotel2_homestay_61to70 ': hotel2_homestay_61to70,
                'hotel2_homestay_71to80 ': hotel2_homestay_71to80,
                'hotel2_homestay_81to90 ': hotel2_homestay_81to90,
                'hotel2_homestay_91to100': hotel2_homestay_91to100,
                'hotel2_homestay_101to110 ': hotel2_homestay_101to110,
                'hotel2_homestay_111to120 ': hotel2_homestay_111to120,
                'hotel2_homestay_121to130 ': hotel2_homestay_121to130,
                'hotel2_homestay_131to140 ': hotel2_homestay_131to140,
                'hotel2_homestay_141to150 ': hotel2_homestay_141to150,
                'hotel2_homestay_151to160 ': hotel2_homestay_151to160,
                'hotel2_homestay_161to170 ': hotel2_homestay_161to170,
                'hotel2_homestay_171to180 ': hotel2_homestay_171to180,
                'hotel2_homestay_181to190 ': hotel2_homestay_181to190,
                'hotel2_homestay_191to200 ': hotel2_homestay_191to200,
                'hotel2_homestay_201to210 ': hotel2_homestay_201to210,
                'hotel2_homestay_211to220': hotel2_homestay_211to220,
                'hotel2_lodge': hotel2_lodge,
                'hotel2_lodge_1to10': hotel2_lodge_1to10,
                'hotel2_lodge_11to20 ': hotel2_lodge_11to20,
                'hotel2_lodge_21to30 ': hotel2_lodge_21to30,
                'hotel2_lodge_31to40 ': hotel2_lodge_31to40,
                'hotel2_lodge_41to50 ': hotel2_lodge_41to50,
                'hotel2_lodge_51to60 ': hotel2_lodge_51to60,
                'hotel2_lodge_61to70 ': hotel2_lodge_61to70,
                'hotel2_lodge_71to80 ': hotel2_lodge_71to80,
                'hotel2_lodge_81to90 ': hotel2_lodge_81to90,
                'hotel2_lodge_91to100': hotel2_lodge_91to100,
                'hotel2_lodge_101to110 ': hotel2_lodge_101to110,
                'hotel2_lodge_111to120 ': hotel2_lodge_111to120,
                'hotel2_lodge_121to130 ': hotel2_lodge_121to130,
                'hotel2_lodge_131to140 ': hotel2_lodge_131to140,
                'hotel2_lodge_141to150 ': hotel2_lodge_141to150,
                'hotel2_lodge_151to160 ': hotel2_lodge_151to160,
                'hotel2_lodge_161to170 ': hotel2_lodge_161to170,
                'hotel2_lodge_171to180 ': hotel2_lodge_171to180,
                'hotel2_lodge_181to190 ': hotel2_lodge_181to190,
                'hotel2_lodge_191to200 ': hotel2_lodge_191to200,
                'hotel2_lodge_201to210 ': hotel2_lodge_201to210,
                'hotel2_lodge_211to220': hotel2_lodge_211to220,
                'hotel2_countryhouse': hotel2_countryhouse,
                'hotel2_countryhouse_1to10': hotel2_countryhouse_1to10,
                'hotel2_countryhouse_11to20 ': hotel2_countryhouse_11to20,
                'hotel2_countryhouse_21to30 ': hotel2_countryhouse_21to30,
                'hotel2_countryhouse_31to40 ': hotel2_countryhouse_31to40,
                'hotel2_countryhouse_41to50 ': hotel2_countryhouse_41to50,
                'hotel2_countryhouse_51to60 ': hotel2_countryhouse_51to60,
                'hotel2_countryhouse_61to70 ': hotel2_countryhouse_61to70,
                'hotel2_countryhouse_71to80 ': hotel2_countryhouse_71to80,
                'hotel2_countryhouse_81to90 ': hotel2_countryhouse_81to90,
                'hotel2_countryhouse_91to100': hotel2_countryhouse_91to100,
                'hotel2_countryhouse_101to110 ': hotel2_countryhouse_101to110,
                'hotel2_countryhouse_111to120 ': hotel2_countryhouse_111to120,
                'hotel2_countryhouse_121to130 ': hotel2_countryhouse_121to130,
                'hotel2_countryhouse_131to140 ': hotel2_countryhouse_131to140,
                'hotel2_countryhouse_141to150 ': hotel2_countryhouse_141to150,
                'hotel2_countryhouse_151to160 ': hotel2_countryhouse_151to160,
                'hotel2_countryhouse_161to170 ': hotel2_countryhouse_161to170,
                'hotel2_countryhouse_171to180 ': hotel2_countryhouse_171to180,
                'hotel2_countryhouse_181to190 ': hotel2_countryhouse_181to190,
                'hotel2_countryhouse_191to200 ': hotel2_countryhouse_191to200,
                'hotel2_countryhouse_201to210 ': hotel2_countryhouse_201to210,
                'hotel2_countryhouse_211to220': hotel2_countryhouse_211to220,
                'hotel2_inn': hotel2_inn,
                'hotel2_inn_1to10': hotel2_inn_1to10,
                'hotel2_inn_11to20 ': hotel2_inn_11to20,
                'hotel2_inn_21to30 ': hotel2_inn_21to30,
                'hotel2_inn_31to40 ': hotel2_inn_31to40,
                'hotel2_inn_41to50 ': hotel2_inn_41to50,
                'hotel2_inn_51to60 ': hotel2_inn_51to60,
                'hotel2_inn_61to70 ': hotel2_inn_61to70,
                'hotel2_inn_71to80 ': hotel2_inn_71to80,
                'hotel2_inn_81to90 ': hotel2_inn_81to90,
                'hotel2_inn_91to100': hotel2_inn_91to100,
                'hotel2_inn_101to110 ': hotel2_inn_101to110,
                'hotel2_inn_111to120 ': hotel2_inn_111to120,
                'hotel2_inn_121to130 ': hotel2_inn_121to130,
                'hotel2_inn_131to140 ': hotel2_inn_131to140,
                'hotel2_inn_141to150 ': hotel2_inn_141to150,
                'hotel2_inn_151to160 ': hotel2_inn_151to160,
                'hotel2_inn_161to170 ': hotel2_inn_161to170,
                'hotel2_inn_171to180 ': hotel2_inn_171to180,
                'hotel2_inn_181to190 ': hotel2_inn_181to190,
                'hotel2_inn_191to200 ': hotel2_inn_191to200,
                'hotel2_inn_201to210 ': hotel2_inn_201to210,
                'hotel2_inn_211to220': hotel2_inn_211to220,
                'hotel2_villa': hotel2_villa,
                'hotel2_villa_1to10': hotel2_villa_1to10,
                'hotel2_villa_11to20 ': hotel2_villa_11to20,
                'hotel2_villa_21to30 ': hotel2_villa_21to30,
                'hotel2_villa_31to40 ': hotel2_villa_31to40,
                'hotel2_villa_41to50 ': hotel2_villa_41to50,
                'hotel2_villa_51to60 ': hotel2_villa_51to60,
                'hotel2_villa_61to70 ': hotel2_villa_61to70,
                'hotel2_villa_71to80 ': hotel2_villa_71to80,
                'hotel2_villa_81to90 ': hotel2_villa_81to90,
                'hotel2_villa_91to100': hotel2_villa_91to100,
                'hotel2_villa_101to110 ': hotel2_villa_101to110,
                'hotel2_villa_111to120 ': hotel2_villa_111to120,
                'hotel2_villa_121to130 ': hotel2_villa_121to130,
                'hotel2_villa_131to140 ': hotel2_villa_131to140,
                'hotel2_villa_141to150 ': hotel2_villa_141to150,
                'hotel2_villa_151to160 ': hotel2_villa_151to160,
                'hotel2_villa_161to170 ': hotel2_villa_161to170,
                'hotel2_villa_171to180 ': hotel2_villa_171to180,
                'hotel2_villa_181to190 ': hotel2_villa_181to190,
                'hotel2_villa_191to200 ': hotel2_villa_191to200,
                'hotel2_villa_201to210 ': hotel2_villa_201to210,
                'hotel2_villa_211to220': hotel2_villa_211to220,
                'hotel2_camping': hotel2_camping,
                'hotel2_camping_1to10': hotel2_camping_1to10,
                'hotel2_camping_11to20 ': hotel2_camping_11to20,
                'hotel2_camping_21to30 ': hotel2_camping_21to30,
                'hotel2_camping_31to40 ': hotel2_camping_31to40,
                'hotel2_camping_41to50 ': hotel2_camping_41to50,
                'hotel2_camping_51to60 ': hotel2_camping_51to60,
                'hotel2_camping_61to70 ': hotel2_camping_61to70,
                'hotel2_camping_71to80 ': hotel2_camping_71to80,
                'hotel2_camping_81to90 ': hotel2_camping_81to90,
                'hotel2_camping_91to100': hotel2_camping_91to100,
                'hotel2_camping_101to110 ': hotel2_camping_101to110,
                'hotel2_camping_111to120 ': hotel2_camping_111to120,
                'hotel2_camping_121to130 ': hotel2_camping_121to130,
                'hotel2_camping_131to140 ': hotel2_camping_131to140,
                'hotel2_camping_141to150 ': hotel2_camping_141to150,
                'hotel2_camping_151to160 ': hotel2_camping_151to160,
                'hotel2_camping_161to170 ': hotel2_camping_161to170,
                'hotel2_camping_171to180 ': hotel2_camping_171to180,
                'hotel2_camping_181to190 ': hotel2_camping_181to190,
                'hotel2_camping_191to200 ': hotel2_camping_191to200,
                'hotel2_camping_201to210 ': hotel2_camping_201to210,
                'hotel2_camping_211to220': hotel2_camping_211to220,
                'hotels3': hotel3,
                'hotel3_1to10 ': hotel3_1to10,
                'hotel3_11to20': hotel3_11to20,
                'hotel3_21to30': hotel3_21to30,
                'hotel3_31to40': hotel3_31to40,
                'hotel3_41to50': hotel3_41to50,
                'hotel3_51to60': hotel3_51to60,
                'hotel3_61to70': hotel3_61to70,
                'hotel3_71to80': hotel3_71to80,
                'hotel3_81to90': hotel3_81to90,
                'hotel3_91to100': hotel3_91to100,
                'hotel3_101to110': hotel3_101to110,
                'hotel3_111to120': hotel3_111to120,
                'hotel3_121to130': hotel3_121to130,
                'hotel3_131to140': hotel3_131to140,
                'hotel3_141to150': hotel3_141to150,
                'hotel3_151to160': hotel3_151to160,
                'hotel3_161to170': hotel3_161to170,
                'hotel3_171to180': hotel3_171to180,
                'hotel3_181to190': hotel3_181to190,
                'hotel3_191to200': hotel3_191to200,
                'hotel3_201to210': hotel3_201to210,
                'hotel3_211to220': hotel3_211to220,
                'hotel3_cost_up': hotel3_cost_up,
                'hotel3_cost_up_1to10': hotel3_cost_up_1to10,
                'hotel3_cost_up_11to20 ': hotel3_cost_up_11to20,
                'hotel3_cost_up_21to30 ': hotel3_cost_up_21to30,
                'hotel3_cost_up_31to40 ': hotel3_cost_up_31to40,
                'hotel3_cost_up_41to50 ': hotel3_cost_up_41to50,
                'hotel3_cost_up_51to60 ': hotel3_cost_up_51to60,
                'hotel3_cost_up_61to70 ': hotel3_cost_up_61to70,
                'hotel3_cost_up_71to80 ': hotel3_cost_up_71to80,
                'hotel3_cost_up_81to90 ': hotel3_cost_up_81to90,
                'hotel3_cost_up_91to100': hotel3_cost_up_91to100,
                'hotel3_cost_up_101to110': hotel3_cost_up_101to110,
                'hotel3_cost_up_111to120': hotel3_cost_up_111to120,
                'hotel3_cost_up_121to130': hotel3_cost_up_121to130,
                'hotel3_cost_up_131to140': hotel3_cost_up_131to140,
                'hotel3_cost_up_141to150': hotel3_cost_up_141to150,
                'hotel3_cost_up_151to160': hotel3_cost_up_151to160,
                'hotel3_cost_up_161to170': hotel3_cost_up_161to170,
                'hotel3_cost_up_171to180': hotel3_cost_up_171to180,
                'hotel3_cost_up_181to190': hotel3_cost_up_181to190,
                'hotel3_cost_up_191to200': hotel3_cost_up_191to200,
                'hotel3_cost_up_201to210': hotel3_cost_up_201to210,
                'hotel3_cost_up_211to220': hotel3_cost_up_211to220,
                'hotel3_cost_down': hotel3_cost_down,
                'hotel3_cost_down_1to10': hotel3_cost_down_1to10,
                'hotel3_cost_down_11to20 ': hotel3_cost_down_11to20,
                'hotel3_cost_down_21to30 ': hotel3_cost_down_21to30,
                'hotel3_cost_down_31to40 ': hotel3_cost_down_31to40,
                'hotel3_cost_down_41to50 ': hotel3_cost_down_41to50,
                'hotel3_cost_down_51to60 ': hotel3_cost_down_51to60,
                'hotel3_cost_down_61to70 ': hotel3_cost_down_61to70,
                'hotel3_cost_down_71to80 ': hotel3_cost_down_71to80,
                'hotel3_cost_down_81to90 ': hotel3_cost_down_81to90,
                'hotel3_cost_down_91to100': hotel3_cost_down_91to100,
                'hotel3_cost_down_101to110': hotel3_cost_down_101to110,
                'hotel3_cost_down_111to120': hotel3_cost_down_111to120,
                'hotel3_cost_down_121to130': hotel3_cost_down_121to130,
                'hotel3_cost_down_131to140': hotel3_cost_down_131to140,
                'hotel3_cost_down_141to150': hotel3_cost_down_141to150,
                'hotel3_cost_down_151to160': hotel3_cost_down_151to160,
                'hotel3_cost_down_161to170': hotel3_cost_down_161to170,
                'hotel3_cost_down_171to180': hotel3_cost_down_171to180,
                'hotel3_cost_down_181to190': hotel3_cost_down_181to190,
                'hotel3_cost_down_191to200': hotel3_cost_down_191to200,
                'hotel3_cost_down_201to210': hotel3_cost_down_201to210,
                'hotel3_cost_down_211to220': hotel3_cost_down_211to220,
                'hotel3_rating_down': hotel3_rating_down,
                'hotel3_rating_down_1to10': hotel3_rating_down_1to10,
                'hotel3_rating_down_11to20 ': hotel3_rating_down_11to20,
                'hotel3_rating_down_21to30 ': hotel3_rating_down_21to30,
                'hotel3_rating_down_31to40 ': hotel3_rating_down_31to40,
                'hotel3_rating_down_41to50 ': hotel3_rating_down_41to50,
                'hotel3_rating_down_51to60 ': hotel3_rating_down_51to60,
                'hotel3_rating_down_61to70 ': hotel3_rating_down_61to70,
                'hotel3_rating_down_71to80 ': hotel3_rating_down_71to80,
                'hotel3_rating_down_81to90 ': hotel3_rating_down_81to90,
                'hotel3_rating_down_91to100': hotel3_rating_down_91to100,
                'hotel3_rating_down_101to110': hotel3_rating_down_101to110,
                'hotel3_rating_down_111to120': hotel3_rating_down_111to120,
                'hotel3_rating_down_121to130': hotel3_rating_down_121to130,
                'hotel3_rating_down_131to140': hotel3_rating_down_131to140,
                'hotel3_rating_down_141to150': hotel3_rating_down_141to150,
                'hotel3_rating_down_151to160': hotel3_rating_down_151to160,
                'hotel3_rating_down_161to170': hotel3_rating_down_161to170,
                'hotel3_rating_down_171to180': hotel3_rating_down_171to180,
                'hotel3_rating_down_181to190': hotel3_rating_down_181to190,
                'hotel3_rating_down_191to200': hotel3_rating_down_191to200,
                'hotel3_rating_down_201to210': hotel3_rating_down_201to210,
                'hotel3_rating_down_211to220': hotel3_rating_down_211to220,
                'hotel3_distance_up': hotel3_distance_up,
                'hotel3_distance_up_1to10': hotel3_distance_up_1to10,
                'hotel3_distance_up_11to20 ': hotel3_distance_up_11to20,
                'hotel3_distance_up_21to30 ': hotel3_distance_up_21to30,
                'hotel3_distance_up_31to40 ': hotel3_distance_up_31to40,
                'hotel3_distance_up_41to50 ': hotel3_distance_up_41to50,
                'hotel3_distance_up_51to60 ': hotel3_distance_up_51to60,
                'hotel3_distance_up_61to70 ': hotel3_distance_up_61to70,
                'hotel3_distance_up_71to80 ': hotel3_distance_up_71to80,
                'hotel3_distance_up_81to90 ': hotel3_distance_up_81to90,
                'hotel3_distance_up_91to100': hotel3_distance_up_91to100,
                'hotel3_distance_up_101to110 ': hotel3_distance_up_101to110,
                'hotel3_distance_up_111to120 ': hotel3_distance_up_111to120,
                'hotel3_distance_up_121to130 ': hotel3_distance_up_121to130,
                'hotel3_distance_up_131to140 ': hotel3_distance_up_131to140,
                'hotel3_distance_up_141to150 ': hotel3_distance_up_141to150,
                'hotel3_distance_up_151to160 ': hotel3_distance_up_151to160,
                'hotel3_distance_up_161to170 ': hotel3_distance_up_161to170,
                'hotel3_distance_up_171to180 ': hotel3_distance_up_171to180,
                'hotel3_distance_up_181to190 ': hotel3_distance_up_181to190,
                'hotel3_distance_up_191to200 ': hotel3_distance_up_191to200,
                'hotel3_distance_up_201to210 ': hotel3_distance_up_201to210,
                'hotel3_distance_up_211to220': hotel3_distance_up_211to220,
                'hotel3_kind_down_1to10': hotel3_kind_down_1to10,
                'hotel3_kind_down_11to20 ': hotel3_kind_down_11to20,
                'hotel3_kind_down_21to30 ': hotel3_kind_down_21to30,
                'hotel3_kind_down_31to40 ': hotel3_kind_down_31to40,
                'hotel3_kind_down_41to50 ': hotel3_kind_down_41to50,
                'hotel3_kind_down_51to60 ': hotel3_kind_down_51to60,
                'hotel3_kind_down_61to70 ': hotel3_kind_down_61to70,
                'hotel3_kind_down_71to80 ': hotel3_kind_down_71to80,
                'hotel3_kind_down_81to90 ': hotel3_kind_down_81to90,
                'hotel3_kind_down_91to100': hotel3_kind_down_91to100,
                'hotel3_kind_down_101to110 ': hotel3_kind_down_101to110,
                'hotel3_kind_down_111to120 ': hotel3_kind_down_111to120,
                'hotel3_kind_down_121to130 ': hotel3_kind_down_121to130,
                'hotel3_kind_down_131to140 ': hotel3_kind_down_131to140,
                'hotel3_kind_down_141to150 ': hotel3_kind_down_141to150,
                'hotel3_kind_down_151to160 ': hotel3_kind_down_151to160,
                'hotel3_kind_down_161to170 ': hotel3_kind_down_161to170,
                'hotel3_kind_down_171to180 ': hotel3_kind_down_171to180,
                'hotel3_kind_down_181to190 ': hotel3_kind_down_181to190,
                'hotel3_kind_down_191to200 ': hotel3_kind_down_191to200,
                'hotel3_kind_down_201to210 ': hotel3_kind_down_201to210,
                'hotel3_kind_down_211to220': hotel3_kind_down_211to220,
                'hotel3_clean_down_1to10': hotel3_clean_down_1to10,
                'hotel3_clean_down_11to20 ': hotel3_clean_down_11to20,
                'hotel3_clean_down_21to30 ': hotel3_clean_down_21to30,
                'hotel3_clean_down_31to40 ': hotel3_clean_down_31to40,
                'hotel3_clean_down_41to50 ': hotel3_clean_down_41to50,
                'hotel3_clean_down_51to60 ': hotel3_clean_down_51to60,
                'hotel3_clean_down_61to70 ': hotel3_clean_down_61to70,
                'hotel3_clean_down_71to80 ': hotel3_clean_down_71to80,
                'hotel3_clean_down_81to90 ': hotel3_clean_down_81to90,
                'hotel3_clean_down_91to100': hotel3_clean_down_91to100,
                'hotel3_clean_down_101to110 ': hotel3_clean_down_101to110,
                'hotel3_clean_down_111to120 ': hotel3_clean_down_111to120,
                'hotel3_clean_down_121to130 ': hotel3_clean_down_121to130,
                'hotel3_clean_down_131to140 ': hotel3_clean_down_131to140,
                'hotel3_clean_down_141to150 ': hotel3_clean_down_141to150,
                'hotel3_clean_down_151to160 ': hotel3_clean_down_151to160,
                'hotel3_clean_down_161to170 ': hotel3_clean_down_161to170,
                'hotel3_clean_down_171to180 ': hotel3_clean_down_171to180,
                'hotel3_clean_down_181to190 ': hotel3_clean_down_181to190,
                'hotel3_clean_down_191to200 ': hotel3_clean_down_191to200,
                'hotel3_clean_down_201to210 ': hotel3_clean_down_201to210,
                'hotel3_clean_down_211to220': hotel3_clean_down_211to220,
                'hotel3_conv_down_1to10': hotel3_conv_down_1to10,
                'hotel3_conv_down_11to20 ': hotel3_conv_down_11to20,
                'hotel3_conv_down_21to30 ': hotel3_conv_down_21to30,
                'hotel3_conv_down_31to40 ': hotel3_conv_down_31to40,
                'hotel3_conv_down_41to50 ': hotel3_conv_down_41to50,
                'hotel3_conv_down_51to60 ': hotel3_conv_down_51to60,
                'hotel3_conv_down_61to70 ': hotel3_conv_down_61to70,
                'hotel3_conv_down_71to80 ': hotel3_conv_down_71to80,
                'hotel3_conv_down_81to90 ': hotel3_conv_down_81to90,
                'hotel3_conv_down_91to100': hotel3_conv_down_91to100,
                'hotel3_conv_down_101to110 ': hotel3_conv_down_101to110,
                'hotel3_conv_down_111to120 ': hotel3_conv_down_111to120,
                'hotel3_conv_down_121to130 ': hotel3_conv_down_121to130,
                'hotel3_conv_down_131to140 ': hotel3_conv_down_131to140,
                'hotel3_conv_down_141to150 ': hotel3_conv_down_141to150,
                'hotel3_conv_down_151to160 ': hotel3_conv_down_151to160,
                'hotel3_conv_down_161to170 ': hotel3_conv_down_161to170,
                'hotel3_conv_down_171to180 ': hotel3_conv_down_171to180,
                'hotel3_conv_down_181to190 ': hotel3_conv_down_181to190,
                'hotel3_conv_down_191to200 ': hotel3_conv_down_191to200,
                'hotel3_conv_down_201to210 ': hotel3_conv_down_201to210,
                'hotel3_conv_down_211to220': hotel3_conv_down_211to220,
                'hotel3_hotel': hotel3_hotel,
                'hotel3_hotel_1to10': hotel3_hotel_1to10,
                'hotel3_hotel_11to20 ': hotel3_hotel_11to20,
                'hotel3_hotel_21to30 ': hotel3_hotel_21to30,
                'hotel3_hotel_31to40 ': hotel3_hotel_31to40,
                'hotel3_hotel_41to50 ': hotel3_hotel_41to50,
                'hotel3_hotel_51to60 ': hotel3_hotel_51to60,
                'hotel3_hotel_61to70 ': hotel3_hotel_61to70,
                'hotel3_hotel_71to80 ': hotel3_hotel_71to80,
                'hotel3_hotel_81to90 ': hotel3_hotel_81to90,
                'hotel3_hotel_91to100': hotel3_hotel_91to100,
                'hotel3_hotel_101to110 ': hotel3_hotel_101to110,
                'hotel3_hotel_111to120 ': hotel3_hotel_111to120,
                'hotel3_hotel_121to130 ': hotel3_hotel_121to130,
                'hotel3_hotel_131to140 ': hotel3_hotel_131to140,
                'hotel3_hotel_141to150 ': hotel3_hotel_141to150,
                'hotel3_hotel_151to160 ': hotel3_hotel_151to160,
                'hotel3_hotel_161to170 ': hotel3_hotel_161to170,
                'hotel3_hotel_171to180 ': hotel3_hotel_171to180,
                'hotel3_hotel_181to190 ': hotel3_hotel_181to190,
                'hotel3_hotel_191to200 ': hotel3_hotel_191to200,
                'hotel3_hotel_201to210 ': hotel3_hotel_201to210,
                'hotel3_hotel_211to220': hotel3_hotel_211to220,
                'hotel3_hostel': hotel3_hostel,
                'hotel3_hostel_1to10': hotel3_hostel_1to10,
                'hotel3_hostel_11to20 ': hotel3_hostel_11to20,
                'hotel3_hostel_21to30 ': hotel3_hostel_21to30,
                'hotel3_hostel_31to40 ': hotel3_hostel_31to40,
                'hotel3_hostel_41to50 ': hotel3_hostel_41to50,
                'hotel3_hostel_51to60 ': hotel3_hostel_51to60,
                'hotel3_hostel_61to70 ': hotel3_hostel_61to70,
                'hotel3_hostel_71to80 ': hotel3_hostel_71to80,
                'hotel3_hostel_81to90 ': hotel3_hostel_81to90,
                'hotel3_hostel_91to100': hotel3_hostel_91to100,
                'hotel3_hostel_101to110 ': hotel3_hostel_101to110,
                'hotel3_hostel_111to120 ': hotel3_hostel_111to120,
                'hotel3_hostel_121to130 ': hotel3_hostel_121to130,
                'hotel3_hostel_131to140 ': hotel3_hostel_131to140,
                'hotel3_hostel_141to150 ': hotel3_hostel_141to150,
                'hotel3_hostel_151to160 ': hotel3_hostel_151to160,
                'hotel3_hostel_161to170 ': hotel3_hostel_161to170,
                'hotel3_hostel_171to180 ': hotel3_hostel_171to180,
                'hotel3_hostel_181to190 ': hotel3_hostel_181to190,
                'hotel3_hostel_191to200 ': hotel3_hostel_191to200,
                'hotel3_hostel_201to210 ': hotel3_hostel_201to210,
                'hotel3_hostel_211to220': hotel3_hostel_211to220,
                'hotel3_guest': hotel3_guest,
                'hotel3_guest_1to10': hotel3_guest_1to10,
                'hotel3_guest_11to20 ': hotel3_guest_11to20,
                'hotel3_guest_21to30 ': hotel3_guest_21to30,
                'hotel3_guest_31to40 ': hotel3_guest_31to40,
                'hotel3_guest_41to50 ': hotel3_guest_41to50,
                'hotel3_guest_51to60 ': hotel3_guest_51to60,
                'hotel3_guest_61to70 ': hotel3_guest_61to70,
                'hotel3_guest_71to80 ': hotel3_guest_71to80,
                'hotel3_guest_81to90 ': hotel3_guest_81to90,
                'hotel3_guest_91to100': hotel3_guest_91to100,
                'hotel3_guest_101to110 ': hotel3_guest_101to110,
                'hotel3_guest_111to120 ': hotel3_guest_111to120,
                'hotel3_guest_121to130 ': hotel3_guest_121to130,
                'hotel3_guest_131to140 ': hotel3_guest_131to140,
                'hotel3_guest_141to150 ': hotel3_guest_141to150,
                'hotel3_guest_151to160 ': hotel3_guest_151to160,
                'hotel3_guest_161to170 ': hotel3_guest_161to170,
                'hotel3_guest_171to180 ': hotel3_guest_171to180,
                'hotel3_guest_181to190 ': hotel3_guest_181to190,
                'hotel3_guest_191to200 ': hotel3_guest_191to200,
                'hotel3_guest_201to210 ': hotel3_guest_201to210,
                'hotel3_guest_211to220': hotel3_guest_211to220,
                'hotel3_apartment': hotel3_apartment,
                'hotel3_apartment_1to10': hotel3_apartment_1to10,
                'hotel3_apartment_11to20 ': hotel3_apartment_11to20,
                'hotel3_apartment_21to30 ': hotel3_apartment_21to30,
                'hotel3_apartment_31to40 ': hotel3_apartment_31to40,
                'hotel3_apartment_41to50 ': hotel3_apartment_41to50,
                'hotel3_apartment_51to60 ': hotel3_apartment_51to60,
                'hotel3_apartment_61to70 ': hotel3_apartment_61to70,
                'hotel3_apartment_71to80 ': hotel3_apartment_71to80,
                'hotel3_apartment_81to90 ': hotel3_apartment_81to90,
                'hotel3_apartment_91to100': hotel3_apartment_91to100,
                'hotel3_apartment_101to110 ': hotel3_apartment_101to110,
                'hotel3_apartment_111to120 ': hotel3_apartment_111to120,
                'hotel3_apartment_121to130 ': hotel3_apartment_121to130,
                'hotel3_apartment_131to140 ': hotel3_apartment_131to140,
                'hotel3_apartment_141to150 ': hotel3_apartment_141to150,
                'hotel3_apartment_151to160 ': hotel3_apartment_151to160,
                'hotel3_apartment_161to170 ': hotel3_apartment_161to170,
                'hotel3_apartment_171to180 ': hotel3_apartment_171to180,
                'hotel3_apartment_181to190 ': hotel3_apartment_181to190,
                'hotel3_apartment_191to200 ': hotel3_apartment_191to200,
                'hotel3_apartment_201to210 ': hotel3_apartment_201to210,
                'hotel3_apartment_211to220': hotel3_apartment_211to220,
                'hotel3_apartmenthotel': hotel3_apartmenthotel,
                'hotel3_apartmenthotel_1to10': hotel3_apartmenthotel_1to10,
                'hotel3_apartmenthotel_11to20 ': hotel3_apartmenthotel_11to20,
                'hotel3_apartmenthotel_21to30 ': hotel3_apartmenthotel_21to30,
                'hotel3_apartmenthotel_31to40 ': hotel3_apartmenthotel_31to40,
                'hotel3_apartmenthotel_41to50 ': hotel3_apartmenthotel_41to50,
                'hotel3_apartmenthotel_51to60 ': hotel3_apartmenthotel_51to60,
                'hotel3_apartmenthotel_61to70 ': hotel3_apartmenthotel_61to70,
                'hotel3_apartmenthotel_71to80 ': hotel3_apartmenthotel_71to80,
                'hotel3_apartmenthotel_81to90 ': hotel3_apartmenthotel_81to90,
                'hotel3_apartmenthotel_91to100': hotel3_apartmenthotel_91to100,
                'hotel3_apartmenthotel_101to110 ':
                hotel3_apartmenthotel_101to110,
                'hotel3_apartmenthotel_111to120 ':
                hotel3_apartmenthotel_111to120,
                'hotel3_apartmenthotel_121to130 ':
                hotel3_apartmenthotel_121to130,
                'hotel3_apartmenthotel_131to140 ':
                hotel3_apartmenthotel_131to140,
                'hotel3_apartmenthotel_141to150 ':
                hotel3_apartmenthotel_141to150,
                'hotel3_apartmenthotel_151to160 ':
                hotel3_apartmenthotel_151to160,
                'hotel3_apartmenthotel_161to170 ':
                hotel3_apartmenthotel_161to170,
                'hotel3_apartmenthotel_171to180 ':
                hotel3_apartmenthotel_171to180,
                'hotel3_apartmenthotel_181to190 ':
                hotel3_apartmenthotel_181to190,
                'hotel3_apartmenthotel_191to200 ':
                hotel3_apartmenthotel_191to200,
                'hotel3_apartmenthotel_201to210 ':
                hotel3_apartmenthotel_201to210,
                'hotel3_apartmenthotel_211to220':
                hotel3_apartmenthotel_211to220,
                'hotel3_motel': hotel3_motel,
                'hotel3_motel_1to10': hotel3_motel_1to10,
                'hotel3_motel_11to20 ': hotel3_motel_11to20,
                'hotel3_motel_21to30 ': hotel3_motel_21to30,
                'hotel3_motel_31to40 ': hotel3_motel_31to40,
                'hotel3_motel_41to50 ': hotel3_motel_41to50,
                'hotel3_motel_51to60 ': hotel3_motel_51to60,
                'hotel3_motel_61to70 ': hotel3_motel_61to70,
                'hotel3_motel_71to80 ': hotel3_motel_71to80,
                'hotel3_motel_81to90 ': hotel3_motel_81to90,
                'hotel3_motel_91to100': hotel3_motel_91to100,
                'hotel3_motel_101to110 ': hotel3_motel_101to110,
                'hotel3_motel_111to120 ': hotel3_motel_111to120,
                'hotel3_motel_121to130 ': hotel3_motel_121to130,
                'hotel3_motel_131to140 ': hotel3_motel_131to140,
                'hotel3_motel_141to150 ': hotel3_motel_141to150,
                'hotel3_motel_151to160 ': hotel3_motel_151to160,
                'hotel3_motel_161to170 ': hotel3_motel_161to170,
                'hotel3_motel_171to180 ': hotel3_motel_171to180,
                'hotel3_motel_181to190 ': hotel3_motel_181to190,
                'hotel3_motel_191to200 ': hotel3_motel_191to200,
                'hotel3_motel_201to210 ': hotel3_motel_201to210,
                'hotel3_motel_211to220': hotel3_motel_211to220,
                'hotel3_pension': hotel3_pension,
                'hotel3_pension_1to10': hotel3_pension_1to10,
                'hotel3_pension_11to20 ': hotel3_pension_11to20,
                'hotel3_pension_21to30 ': hotel3_pension_21to30,
                'hotel3_pension_31to40 ': hotel3_pension_31to40,
                'hotel3_pension_41to50 ': hotel3_pension_41to50,
                'hotel3_pension_51to60 ': hotel3_pension_51to60,
                'hotel3_pension_61to70 ': hotel3_pension_61to70,
                'hotel3_pension_71to80 ': hotel3_pension_71to80,
                'hotel3_pension_81to90 ': hotel3_pension_81to90,
                'hotel3_pension_91to100': hotel3_pension_91to100,
                'hotel3_pension_101to110 ': hotel3_pension_101to110,
                'hotel3_pension_111to120 ': hotel3_pension_111to120,
                'hotel3_pension_121to130 ': hotel3_pension_121to130,
                'hotel3_pension_131to140 ': hotel3_pension_131to140,
                'hotel3_pension_141to150 ': hotel3_pension_141to150,
                'hotel3_pension_151to160 ': hotel3_pension_151to160,
                'hotel3_pension_161to170 ': hotel3_pension_161to170,
                'hotel3_pension_171to180 ': hotel3_pension_171to180,
                'hotel3_pension_181to190 ': hotel3_pension_181to190,
                'hotel3_pension_191to200 ': hotel3_pension_191to200,
                'hotel3_pension_201to210 ': hotel3_pension_201to210,
                'hotel3_pension_211to220': hotel3_pension_211to220,
                'hotel3_resort': hotel3_resort,
                'hotel3_resort_1to10': hotel3_resort_1to10,
                'hotel3_resort_11to20 ': hotel3_resort_11to20,
                'hotel3_resort_21to30 ': hotel3_resort_21to30,
                'hotel3_resort_31to40 ': hotel3_resort_31to40,
                'hotel3_resort_41to50 ': hotel3_resort_41to50,
                'hotel3_resort_51to60 ': hotel3_resort_51to60,
                'hotel3_resort_61to70 ': hotel3_resort_61to70,
                'hotel3_resort_71to80 ': hotel3_resort_71to80,
                'hotel3_resort_81to90 ': hotel3_resort_81to90,
                'hotel3_resort_91to100': hotel3_resort_91to100,
                'hotel3_resort_101to110 ': hotel3_resort_101to110,
                'hotel3_resort_111to120 ': hotel3_resort_111to120,
                'hotel3_resort_121to130 ': hotel3_resort_121to130,
                'hotel3_resort_131to140 ': hotel3_resort_131to140,
                'hotel3_resort_141to150 ': hotel3_resort_141to150,
                'hotel3_resort_151to160 ': hotel3_resort_151to160,
                'hotel3_resort_161to170 ': hotel3_resort_161to170,
                'hotel3_resort_171to180 ': hotel3_resort_171to180,
                'hotel3_resort_181to190 ': hotel3_resort_181to190,
                'hotel3_resort_191to200 ': hotel3_resort_191to200,
                'hotel3_resort_201to210 ': hotel3_resort_201to210,
                'hotel3_resort_211to220': hotel3_resort_211to220,
                'hotel3_badandbreakfast': hotel3_badandbreakfast,
                'hotel3_badandbreakfast_1to10': hotel3_badandbreakfast_1to10,
                'hotel3_badandbreakfast_11to20 ':
                hotel3_badandbreakfast_11to20,
                'hotel3_badandbreakfast_21to30 ':
                hotel3_badandbreakfast_21to30,
                'hotel3_badandbreakfast_31to40 ':
                hotel3_badandbreakfast_31to40,
                'hotel3_badandbreakfast_41to50 ':
                hotel3_badandbreakfast_41to50,
                'hotel3_badandbreakfast_51to60 ':
                hotel3_badandbreakfast_51to60,
                'hotel3_badandbreakfast_61to70 ':
                hotel3_badandbreakfast_61to70,
                'hotel3_badandbreakfast_71to80 ':
                hotel3_badandbreakfast_71to80,
                'hotel3_badandbreakfast_81to90 ':
                hotel3_badandbreakfast_81to90,
                'hotel3_badandbreakfast_91to100':
                hotel3_badandbreakfast_91to100,
                'hotel3_badandbreakfast_101to110 ':
                hotel3_badandbreakfast_101to110,
                'hotel3_badandbreakfast_111to120 ':
                hotel3_badandbreakfast_111to120,
                'hotel3_badandbreakfast_121to130 ':
                hotel3_badandbreakfast_121to130,
                'hotel3_badandbreakfast_131to140 ':
                hotel3_badandbreakfast_131to140,
                'hotel3_badandbreakfast_141to150 ':
                hotel3_badandbreakfast_141to150,
                'hotel3_badandbreakfast_151to160 ':
                hotel3_badandbreakfast_151to160,
                'hotel3_badandbreakfast_161to170 ':
                hotel3_badandbreakfast_161to170,
                'hotel3_badandbreakfast_171to180 ':
                hotel3_badandbreakfast_171to180,
                'hotel3_badandbreakfast_181to190 ':
                hotel3_badandbreakfast_181to190,
                'hotel3_badandbreakfast_191to200 ':
                hotel3_badandbreakfast_191to200,
                'hotel3_badandbreakfast_201to210 ':
                hotel3_badandbreakfast_201to210,
                'hotel3_badandbreakfast_211to220':
                hotel3_badandbreakfast_211to220,
                'hotel3_homestay': hotel3_homestay,
                'hotel3_homestay_1to10': hotel3_homestay_1to10,
                'hotel3_homestay_11to20 ': hotel3_homestay_11to20,
                'hotel3_homestay_21to30 ': hotel3_homestay_21to30,
                'hotel3_homestay_31to40 ': hotel3_homestay_31to40,
                'hotel3_homestay_41to50 ': hotel3_homestay_41to50,
                'hotel3_homestay_51to60 ': hotel3_homestay_51to60,
                'hotel3_homestay_61to70 ': hotel3_homestay_61to70,
                'hotel3_homestay_71to80 ': hotel3_homestay_71to80,
                'hotel3_homestay_81to90 ': hotel3_homestay_81to90,
                'hotel3_homestay_91to100': hotel3_homestay_91to100,
                'hotel3_homestay_101to110 ': hotel3_homestay_101to110,
                'hotel3_homestay_111to120 ': hotel3_homestay_111to120,
                'hotel3_homestay_121to130 ': hotel3_homestay_121to130,
                'hotel3_homestay_131to140 ': hotel3_homestay_131to140,
                'hotel3_homestay_141to150 ': hotel3_homestay_141to150,
                'hotel3_homestay_151to160 ': hotel3_homestay_151to160,
                'hotel3_homestay_161to170 ': hotel3_homestay_161to170,
                'hotel3_homestay_171to180 ': hotel3_homestay_171to180,
                'hotel3_homestay_181to190 ': hotel3_homestay_181to190,
                'hotel3_homestay_191to200 ': hotel3_homestay_191to200,
                'hotel3_homestay_201to210 ': hotel3_homestay_201to210,
                'hotel3_homestay_211to220': hotel3_homestay_211to220,
                'hotel3_lodge': hotel3_lodge,
                'hotel3_lodge_1to10': hotel3_lodge_1to10,
                'hotel3_lodge_11to20 ': hotel3_lodge_11to20,
                'hotel3_lodge_21to30 ': hotel3_lodge_21to30,
                'hotel3_lodge_31to40 ': hotel3_lodge_31to40,
                'hotel3_lodge_41to50 ': hotel3_lodge_41to50,
                'hotel3_lodge_51to60 ': hotel3_lodge_51to60,
                'hotel3_lodge_61to70 ': hotel3_lodge_61to70,
                'hotel3_lodge_71to80 ': hotel3_lodge_71to80,
                'hotel3_lodge_81to90 ': hotel3_lodge_81to90,
                'hotel3_lodge_91to100': hotel3_lodge_91to100,
                'hotel3_lodge_101to110 ': hotel3_lodge_101to110,
                'hotel3_lodge_111to120 ': hotel3_lodge_111to120,
                'hotel3_lodge_121to130 ': hotel3_lodge_121to130,
                'hotel3_lodge_131to140 ': hotel3_lodge_131to140,
                'hotel3_lodge_141to150 ': hotel3_lodge_141to150,
                'hotel3_lodge_151to160 ': hotel3_lodge_151to160,
                'hotel3_lodge_161to170 ': hotel3_lodge_161to170,
                'hotel3_lodge_171to180 ': hotel3_lodge_171to180,
                'hotel3_lodge_181to190 ': hotel3_lodge_181to190,
                'hotel3_lodge_191to200 ': hotel3_lodge_191to200,
                'hotel3_lodge_201to210 ': hotel3_lodge_201to210,
                'hotel3_lodge_211to220': hotel3_lodge_211to220,
                'hotel3_countryhouse': hotel3_countryhouse,
                'hotel3_countryhouse_1to10': hotel3_countryhouse_1to10,
                'hotel3_countryhouse_11to20 ': hotel3_countryhouse_11to20,
                'hotel3_countryhouse_21to30 ': hotel3_countryhouse_21to30,
                'hotel3_countryhouse_31to40 ': hotel3_countryhouse_31to40,
                'hotel3_countryhouse_41to50 ': hotel3_countryhouse_41to50,
                'hotel3_countryhouse_51to60 ': hotel3_countryhouse_51to60,
                'hotel3_countryhouse_61to70 ': hotel3_countryhouse_61to70,
                'hotel3_countryhouse_71to80 ': hotel3_countryhouse_71to80,
                'hotel3_countryhouse_81to90 ': hotel3_countryhouse_81to90,
                'hotel3_countryhouse_91to100': hotel3_countryhouse_91to100,
                'hotel3_countryhouse_101to110 ': hotel3_countryhouse_101to110,
                'hotel3_countryhouse_111to120 ': hotel3_countryhouse_111to120,
                'hotel3_countryhouse_121to130 ': hotel3_countryhouse_121to130,
                'hotel3_countryhouse_131to140 ': hotel3_countryhouse_131to140,
                'hotel3_countryhouse_141to150 ': hotel3_countryhouse_141to150,
                'hotel3_countryhouse_151to160 ': hotel3_countryhouse_151to160,
                'hotel3_countryhouse_161to170 ': hotel3_countryhouse_161to170,
                'hotel3_countryhouse_171to180 ': hotel3_countryhouse_171to180,
                'hotel3_countryhouse_181to190 ': hotel3_countryhouse_181to190,
                'hotel3_countryhouse_191to200 ': hotel3_countryhouse_191to200,
                'hotel3_countryhouse_201to210 ': hotel3_countryhouse_201to210,
                'hotel3_countryhouse_211to220': hotel3_countryhouse_211to220,
                'hotel3_inn': hotel3_inn,
                'hotel3_inn_1to10': hotel3_inn_1to10,
                'hotel3_inn_11to20 ': hotel3_inn_11to20,
                'hotel3_inn_21to30 ': hotel3_inn_21to30,
                'hotel3_inn_31to40 ': hotel3_inn_31to40,
                'hotel3_inn_41to50 ': hotel3_inn_41to50,
                'hotel3_inn_51to60 ': hotel3_inn_51to60,
                'hotel3_inn_61to70 ': hotel3_inn_61to70,
                'hotel3_inn_71to80 ': hotel3_inn_71to80,
                'hotel3_inn_81to90 ': hotel3_inn_81to90,
                'hotel3_inn_91to100': hotel3_inn_91to100,
                'hotel3_inn_101to110 ': hotel3_inn_101to110,
                'hotel3_inn_111to120 ': hotel3_inn_111to120,
                'hotel3_inn_121to130 ': hotel3_inn_121to130,
                'hotel3_inn_131to140 ': hotel3_inn_131to140,
                'hotel3_inn_141to150 ': hotel3_inn_141to150,
                'hotel3_inn_151to160 ': hotel3_inn_151to160,
                'hotel3_inn_161to170 ': hotel3_inn_161to170,
                'hotel3_inn_171to180 ': hotel3_inn_171to180,
                'hotel3_inn_181to190 ': hotel3_inn_181to190,
                'hotel3_inn_191to200 ': hotel3_inn_191to200,
                'hotel3_inn_201to210 ': hotel3_inn_201to210,
                'hotel3_inn_211to220': hotel3_inn_211to220,
                'hotel3_villa': hotel3_villa,
                'hotel3_villa_1to10': hotel3_villa_1to10,
                'hotel3_villa_11to20 ': hotel3_villa_11to20,
                'hotel3_villa_21to30 ': hotel3_villa_21to30,
                'hotel3_villa_31to40 ': hotel3_villa_31to40,
                'hotel3_villa_41to50 ': hotel3_villa_41to50,
                'hotel3_villa_51to60 ': hotel3_villa_51to60,
                'hotel3_villa_61to70 ': hotel3_villa_61to70,
                'hotel3_villa_71to80 ': hotel3_villa_71to80,
                'hotel3_villa_81to90 ': hotel3_villa_81to90,
                'hotel3_villa_91to100': hotel3_villa_91to100,
                'hotel3_villa_101to110 ': hotel3_villa_101to110,
                'hotel3_villa_111to120 ': hotel3_villa_111to120,
                'hotel3_villa_121to130 ': hotel3_villa_121to130,
                'hotel3_villa_131to140 ': hotel3_villa_131to140,
                'hotel3_villa_141to150 ': hotel3_villa_141to150,
                'hotel3_villa_151to160 ': hotel3_villa_151to160,
                'hotel3_villa_161to170 ': hotel3_villa_161to170,
                'hotel3_villa_171to180 ': hotel3_villa_171to180,
                'hotel3_villa_181to190 ': hotel3_villa_181to190,
                'hotel3_villa_191to200 ': hotel3_villa_191to200,
                'hotel3_villa_201to210 ': hotel3_villa_201to210,
                'hotel3_villa_211to220': hotel3_villa_211to220,
                'hotel3_camping': hotel3_camping,
                'hotel3_camping_1to10': hotel3_camping_1to10,
                'hotel3_camping_11to20 ': hotel3_camping_11to20,
                'hotel3_camping_21to30 ': hotel3_camping_21to30,
                'hotel3_camping_31to40 ': hotel3_camping_31to40,
                'hotel3_camping_41to50 ': hotel3_camping_41to50,
                'hotel3_camping_51to60 ': hotel3_camping_51to60,
                'hotel3_camping_61to70 ': hotel3_camping_61to70,
                'hotel3_camping_71to80 ': hotel3_camping_71to80,
                'hotel3_camping_81to90 ': hotel3_camping_81to90,
                'hotel3_camping_91to100': hotel3_camping_91to100,
                'hotel3_camping_101to110 ': hotel3_camping_101to110,
                'hotel3_camping_111to120 ': hotel3_camping_111to120,
                'hotel3_camping_121to130 ': hotel3_camping_121to130,
                'hotel3_camping_131to140 ': hotel3_camping_131to140,
                'hotel3_camping_141to150 ': hotel3_camping_141to150,
                'hotel3_camping_151to160 ': hotel3_camping_151to160,
                'hotel3_camping_161to170 ': hotel3_camping_161to170,
                'hotel3_camping_171to180 ': hotel3_camping_171to180,
                'hotel3_camping_181to190 ': hotel3_camping_181to190,
                'hotel3_camping_191to200 ': hotel3_camping_191to200,
                'hotel3_camping_201to210 ': hotel3_camping_201to210,
                'hotel3_camping_211to220': hotel3_camping_211to220,
                # 'restaurant1': restaurant1,
                # 'restaurant2': restaurant2,
                # 'restaurant3': restaurant3,
                # 'restaurant4': restaurant4,
                # 'restaurant5': restaurant5,
            })
    else:
        return render(request, 'beer/ver1.html', context)


# def ver_result(request):
#     beer_list = pd.read_csv('result.csv', encoding='utf-8', index_col=0)

#     if request.method == "POST":
#         if "beer_list" in request.POST:
#             request.session['ver2'] = request.POST["beer_list"]
#         if "beer_list" in requset.session:
#             context['ver2'] = request.seession['ver2']

#     return render(request, ver2.html, context)

# def ver2_save(request):
#     beer_list = pd.read_csv('result.csv', encoding='utf-8', index_col=0)
#     ratings = pd.read_csv('merge.csv', encoding='utf-8', index_col=0)
#     cluster_3 = pd.read_csv('대표군집클러스터링.csv', encoding='utf-8', index_col=0)
#     cluster_all = pd.read_csv('전체도시클러스터링.csv', encoding='utf-8', index_col=0)
#     beer_list = beer_list['locate']

#     if "beer_list" in request.POST:
#         request.session['ver2'] = request.POST["beer_list"]
#         if "beer_list" in requset.session:
#             context['ver2'] = request.seession['ver2']

#             beer_name = context['ver2']

#             df = recomm_feature(ratings)

#             result = recomm_beer(df, beer_name)
#             result = result.index.tolist()

#             hotel1 = Hotel.objects.filter(place=result[0])

#             page = request.GET.get('page', 1)

#             paginator = Paginator(hotel1, 10)
#             posts = paginator.get_page(page)

#         return render(
#             request, ver2_result.html, {
#                 'beer_name': beer_name,
#                 'result': result,
#                 'hotels1': hotel1,
#                 'posts': posts,
#             })


def ver2(request):
    beer_list = pd.read_csv('result.csv', encoding='utf-8', index_col=0)

    beer_list = beer_list['locate']
    
    text = {'beer_list': beer_list}

    login_session = request.session.get('login_session')

    if login_session == '':
        text['login_session'] = False
    else:
        text['login_session'] = True
    if request.method == 'POST':
         beer_name = request.POST.get('beer', '')
         request.session['tour'] = beer_name
         text['tour'] = request.session['tour']

    return render(request, 'beer/ver2.html', text)
        

def ver2_session(request):
    ratings = pd.read_csv('merge.csv', encoding='utf-8', index_col=0)
    cluster_3 = pd.read_csv('대표군집클러스터링.csv', encoding='utf-8', index_col=0)
    cluster_all = pd.read_csv('전체도시클러스터링.csv', encoding='utf-8', index_col=0)

    beer_name = request.session.get('tour')

    df = recomm_feature(ratings)

    result = recomm_beer(df, beer_name)
    result = result.index. tolist()

    login_session = request.session.get('login_session')

    if login_session == '':
        request.session['login_session'] = False
    else:
        request.session['login_session'] = True

    hotel1 = Hotel.objects.filter(place=result[0])
    page = request.GET.get('page', 1)

    paginator = Paginator(hotel1, 10)
    posts = paginator.get_page(page)

    return render(
        request, 'beer/ver2_result.html', {
            'login_session': login_session,
            'result': result,
            'hotels1': hotel1,
            'posts': posts,
        })


# def ver2_result(request):
#     beer_list = pd.read_csv('result.csv', encoding='utf-8', index_col=0)
#     ratings = pd.read_csv('merge.csv', encoding='utf-8', index_col=0)
#     cluster_3 = pd.read_csv('대표군집클러스터링.csv', encoding='utf-8', index_col=0)
#     cluster_all = pd.read_csv('전체도시클러스터링.csv', encoding='utf-8', index_col=0)
#     beer_list = beer_list['locate']

#     # 여기서 POST 방식으로 받아서
#     if request.method == 'POST':

#         beer_name = request.POST.get('beer', '')

#         df = recomm_feature(ratings)

#         result = recomm_beer(df, beer_name)
#         result = result.index.tolist()

#         # 결과페이지에서 로그인 세션 유지
#         login_session = request.session.get('login_session')

#         if login_session == '':
#             request.session['login_session'] = False
#         else:
#             request.session['login_session'] = True

#         # 가격 등급 리뷰개수 평점 거리 필터링
#         hotel1 = Hotel.objects.filter(place=result[0])
#         page = request.GET.get('page', 1)

#         paginator = Paginator(hotel1, 10)
#         posts = paginator.get_page(page)
#         # 고치기

#         return render(
#             request, 'beer/ver2_result.html', {
#                 'login_session': login_session,
#                 'result': result,
#                 'hotels1': hotel1,
#                 'posts': posts,
#             })

#     # 여기는 GET방식으로 돌아오기 때문에 pagination이 GET방식이기 떄문에 여기로 돌아옴
#     else:
#         # 로그인 세션 유지
#         text = {'beer_list': beer_list}
#         login_session = request.session.get('login_session')

#         if login_session == '':
#             text['login_session'] = False
#         else:
#             text['login_session'] = True

#         return render(request, 'beer/ver2.html', text)


def ver3(request):
    df_cluster = pd.read_csv('result.csv', encoding='utf-8', index_col=0)

    cst0_list = df_cluster.loc[df_cluster['Cluster'] == 0, 'place'].tolist()

    cst1_list = df_cluster.loc[df_cluster['Cluster'] == 1, 'place'].tolist()

    cst2_list = df_cluster.loc[df_cluster['Cluster'] == 2, 'place'].tolist()

    cst3_list = df_cluster.loc[df_cluster['Cluster'] == 3, 'place'].tolist()

    cst4_list = df_cluster.loc[df_cluster['Cluster'] == 4, 'place'].tolist()

    cst5_list = df_cluster.loc[df_cluster['Cluster'] == 5, 'place'].tolist()

    cst6_list = df_cluster.loc[df_cluster['Cluster'] == 6, 'place'].tolist()

    cst7_list = df_cluster.loc[df_cluster['Cluster'] == 7, 'place'].tolist()

    cst8_list = df_cluster.loc[df_cluster['Cluster'] == 8, 'place'].tolist()

    cst9_list = df_cluster.loc[df_cluster['Cluster'] == 9, 'place'].tolist()

    cst10_list = df_cluster.loc[df_cluster['Cluster'] == 10, 'place'].tolist()

    cst11_list = df_cluster.loc[df_cluster['Cluster'] == 11, 'place'].tolist()

    # ver3에서 로그인 세션 유지
    context = {}
    login_session = request.session.get('login_session')

    if login_session == '':
        context['login_session'] = False
    else:
        context['login_session'] = True

    if request.method == 'POST':

        # 결과페이지에서 로그인 세션 유지
        login_session = request.session.get('login_session')

        if login_session == '':
            request.session['login_session'] = False
        else:
            request.session['login_session'] = True

        # detail value POST
        detail = request.POST.get('detail', '')
        detail2 = request.POST.get('topic', )
        if detail in ['food', 'walk', 'nature']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'culture']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'date']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'sleep']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'drive']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'night']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'walk', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'culture']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'date']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'sleep']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'drive']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'night']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'nature', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'date']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'sleep']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'drive']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'night']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'culture', 'view']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'sleep']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'drive']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'night']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'date', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'drive']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'night']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'sleep', 'view']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'night']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'sns']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'family']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['food', 'drive', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'night', 'fori']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['food', 'night', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'night', 'family']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['food', 'night', 'view']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['food', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'fori', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['food', 'sns', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['food', 'sns', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['food', 'family', 'view']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'culture']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'date']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'sleep']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'drive']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'night']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'fori']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'nature', 'view']:
            result = cst_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'date']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'sleep']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'drive']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'night']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['walk', 'culture', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'sleep']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'drive']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'sns']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'date', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'drive']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'night']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'sleep', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'fori']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'sns']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['walk', 'drive', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'night', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['walk', 'night', 'family']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'night', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['walk', 'fori', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['walk', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['walk', 'fori', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['walk', 'sns', 'family']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'sns', 'view']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['walk', 'family', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'date']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'sleep']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'drive']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'night']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'sns']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'family']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['nature', 'culture', 'view']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'sleep']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'drive']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'night']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['nature', 'date', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'drive']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'night']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'sns']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'sleep', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'sns']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'drive', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['nature', 'night', 'fori']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['nature', 'night', 'sns']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['nature', 'night', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'night', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'fori', 'family']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['nature', 'fori', 'view']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['nature', 'sns', 'family']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['nature', 'sns', 'view']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['nature', 'family', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'sleep']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'drive']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'night']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'date', 'view']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'drive']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'night']:
            result = cst_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'fori']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'sns']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['culture', 'sleep', 'view']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'night']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'sns']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'family']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['culture', 'drive', 'view']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['culture', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'night', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['culture', 'night', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['culture', 'night', 'view']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['culture', 'fori', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['culture', 'sns', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['culture', 'sns', 'view']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['culture', 'family', 'view']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'drive']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'night']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'sns']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'family']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['date', 'sleep', 'view']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['date', 'drive', 'view']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['date', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['date', 'night', 'sns']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['date', 'night', 'family']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['date', 'night', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['date', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['date', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['date', 'fori', 'view']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['date', 'sns', 'family']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['date', 'sns', 'view']:
            result = cst0_list
            random.shuffle(result)

        elif detail in ['date', 'family', 'view']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'night']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'family']:
            result = cst10_list
            random.shuffle(result)

        elif detail in ['sleep', 'drive', 'view']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['sleep', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['sleep', 'night', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['sleep', 'night', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['sleep', 'night', 'view']:
            result = cst8_list
            random.shuffle(result)

        elif detail in ['sleep', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['sleep', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['sleep', 'fori', 'view']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['sleep', 'sns', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['sleep', 'sns', 'view']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['sleep', 'family', 'view']:
            result = cst9_list
            random.shuffle(result)

        elif detail in ['drive', 'night', 'fori']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['drive', 'night', 'sns']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['drive', 'night', 'family']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['drive', 'night', 'view']:
            result = cst2_list
            random.shuffle(result)

        elif detail in ['drive', 'sns', 'family']:
            result = cst1_list
            random.shuffle(result)

        elif detail in ['drive', 'sns', 'view']:
            result = cst3_list
            random.shuffle(result)

        elif detail in ['drive', 'family', 'view']:
            result = cst7_list
            random.shuffle(result)

        elif detail in ['night', 'fori', 'sns']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['night', 'fori', 'family']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['night', 'fori', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['night', 'sns', 'family']:
            result = cst11_list
            random.shuffle(result)

        elif detail in ['night', 'sns', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['night', 'family', 'view']:
            result = cst4_list
            random.shuffle(result)

        elif detail in ['fori', 'sns', 'family']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['fori', 'sns', 'view']:
            result = cst5_list
            random.shuffle(result)

        elif detail in ['fori', 'family', 'view']:
            result = cst6_list
            random.shuffle(result)

        elif detail in ['sns', 'family', 'view']:
            result = cst2_list
            random.shuffle(result)

        hotel1 = Hotel.objects.filter(place=result[0])
        hotel1_cost_up = hotel1.order_by('cost')
        hotel1_cost_down = hotel1.order_by('-cost')
        hotel1_rating_up = hotel1.order_by('rating')
        hotel1_rating_down = hotel1.order_by('-rating')
        hotel1_distance_up = hotel1.order_by('distance')
        hotel1_kind_up = hotel1.order_by('kind')
        hotel1_clean_up = hotel1.order_by('clean')
        hotel1_conv_up = hotel1.order_by('conv')
        hotel1_hotel = Hotel.objects.filter(place=result[0],
                                            classfication='호텔')
        hotel1_hostel = Hotel.objects.filter(place=result[0],
                                             classfication='호스텔')
        hotel1_guest = Hotel.objects.filter(place=result[0],
                                            classfication='게스트하우스')
        hotel1_apartment = Hotel.objects.filter(place=result[0],
                                                classfication='아파트')
        hotel1_apartmenthotel = Hotel.objects.filter(place=result[0],
                                                     classfication='아파트호텔')
        hotel1_motel = Hotel.objects.filter(place=result[0],
                                            classfication='모텔')
        hotel1_pension = Hotel.objects.filter(place=result[0],
                                              classfication='펜션')
        hotel1_resort = Hotel.objects.filter(place=result[0],
                                             classfication='리조트')
        hotel1_badandbreakfast = Hotel.objects.filter(place=result[0],
                                                      classfication='베드앤브렉퍼스트')
        hotel1_homestay = Hotel.objects.filter(place=result[0],
                                               classfication='홈스테이')
        hotel1_lodge = Hotel.objects.filter(place=result[0],
                                            classfication='롯지')
        hotel1_countryhouse = Hotel.objects.filter(place=result[0],
                                                   classfication='컨트리하우스')
        hotel1_inn = Hotel.objects.filter(place=result[0], classfication='여관')
        hotel1_villa = Hotel.objects.filter(place=result[0],
                                            classfication='빌라')
        hotel1_camping = Hotel.objects.filter(place=result[0],
                                              classfication='캠핑장')

        paginator = Paginator(hotel1, 10)

        posts = paginator.get_page(page)

        hotel2 = Hotel.objects.filter(place=result[1])
        hotel2_cost_up = hotel2.order_by('cost')
        hotel2_cost_down = hotel2.order_by('-cost')
        hotel2_rating_up = hotel2.order_by('rating')
        hotel2_rating_down = hotel2.order_by('-rating')
        hotel2_distance_up = hotel2.order_by('distance')
        hotel2_kind_up = hotel2.order_by('kind')
        hotel2_clean_up = hotel2.order_by('clean')
        hotel2_conv_up = hotel2.order_by('conv')
        hotel2_hotel = Hotel.objects.filter(place=result[1],
                                            classfication='호텔')
        hotel2_hostel = Hotel.objects.filter(place=result[1],
                                             classfication='호스텔')
        hotel2_guest = Hotel.objects.filter(place=result[1],
                                            classfication='게스트하우스')
        hotel2_apartment = Hotel.objects.filter(place=result[1],
                                                classfication='아파트')
        hotel2_apartmenthotel = Hotel.objects.filter(place=result[1],
                                                     classfication='아파트호텔')
        hotel2_motel = Hotel.objects.filter(place=result[1],
                                            classfication='모텔')
        hotel2_pension = Hotel.objects.filter(place=result[1],
                                              classfication='펜션')
        hotel2_resort = Hotel.objects.filter(place=result[1],
                                             classfication='리조트')
        hotel2_badandbreakfast = Hotel.objects.filter(place=result[1],
                                                      classfication='베드앤브렉퍼스트')
        hotel2_homestay = Hotel.objects.filter(place=result[1],
                                               classfication='홈스테이')
        hotel2_lodge = Hotel.objects.filter(place=result[1],
                                            classfication='롯지')
        hotel2_countryhouse = Hotel.objects.filter(place=result[1],
                                                   classfication='컨트리하우스')
        hotel2_inn = Hotel.objects.filter(place=result[1], classfication='여관')
        hotel2_villa = Hotel.objects.filter(place=result[1],
                                            classfication='빌라')
        hotel2_camping = Hotel.objects.filter(place=result[1],
                                              classfication='캠핑장')

        hotel3 = Hotel.objects.filter(place=result[2])
        hotel3_cost_up = hotel3.order_by('cost')
        hotel3_cost_down = hotel3.order_by('-cost')
        hotel3_rating_up = hotel3.order_by('rating')
        hotel3_rating_down = hotel3.order_by('-rating')
        hotel3_distance_up = hotel3.order_by('distance')
        hotel3_kind_up = hotel3.order_by('kind')
        hotel3_clean_up = hotel3.order_by('clean')
        hotel3_conv_up = hotel3.order_by('conv')
        hotel3_hotel = Hotel.objects.filter(place=result[2],
                                            classfication='호텔')
        hotel3_hostel = Hotel.objects.filter(place=result[2],
                                             classfication='호스텔')
        hotel3_guest = Hotel.objects.filter(place=result[2],
                                            classfication='게스트하우스')
        hotel3_apartment = Hotel.objects.filter(place=result[2],
                                                classfication='아파트')
        hotel3_apartmenthotel = Hotel.objects.filter(place=result[2],
                                                     classfication='아파트호텔')
        hotel3_motel = Hotel.objects.filter(place=result[2],
                                            classfication='모텔')
        hotel3_pension = Hotel.objects.filter(place=result[2],
                                              classfication='펜션')
        hotel3_resort = Hotel.objects.filter(place=result[2],
                                             classfication='리조트')
        hotel3_badandbreakfast = Hotel.objects.filter(place=result[2],
                                                      classfication='베드앤브렉퍼스트')
        hotel3_homestay = Hotel.objects.filter(place=result[2],
                                               classfication='홈스테이')
        hotel3_lodge = Hotel.objects.filter(place=result[2],
                                            classfication='롯지')
        hotel3_countryhouse = Hotel.objects.filter(place=result[2],
                                                   classfication='컨트리하우스')
        hotel3_inn = Hotel.objects.filter(place=result[2], classfication='여관')
        hotel3_villa = Hotel.objects.filter(place=result[2],
                                            classfication='빌라')
        hotel3_camping = Hotel.objects.filter(place=result[2],
                                              classfication='캠핑장')

        # restaurant1 = Restaurant.objects.filter(place=result[0])
        # restaurant2 = Restaurant.objects.filter(place=result[1])
        # restaurant3 = Restaurant.objects.filter(place=result[2])
        # restaurant4 = Restaurant.objects.filter(place=result[3])
        # restaurant5 = Restaurant.objects.filter(place=result[4])

        return render(
            request,
            'beer/ver3_result.html',
            {
                'login_session': login_session,
                'result': result,
                'hotels1': hotel1,
                'hotels2': hotel2,
                'hotels3': hotel3,
                'hotels4': hotel4,
                'hotels5': hotel5,
                # 'restaurant1': restaurant1,
                # 'restaurant2': restaurant2,
                # 'restaurant3': restaurant3,
                # 'restaurant4': restaurant4,
                # 'restaurant5': restaurant5,
            })
    else:
        return render(request, 'beer/ver3.html', context)
