from django.urls import path
from . import views

app_name = 'beer'
urlpatterns = [
    path('ver1', views.ver1, name='ver1'),
    path('ver3', views.ver3, name='ver3'),
    path('ver2', views.ver2, name='ver2'),
    path('ver2', views.ver2, name='ver2'),
    path('ver2_session', views.ver2_session, name='ver2_session')
]
