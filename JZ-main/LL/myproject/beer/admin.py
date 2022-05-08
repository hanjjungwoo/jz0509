from django.contrib import admin
from .models import *
# from .models import User

# Register your models here.


class HotelAdmin(admin.ModelAdmin):
    list_display = ('index', 'place', 'name', 'rating', 'distance', 'cost',
                    'address', 'explain', 'kind', 'clean', 'conv', 'url',
                    'img', 'classfication')


# class RestaurantAdmin(admin.ModelAdmin):
#     list_display = ('index', 'locate', 'name', 'rating', 'classfications',
#                     'address', 'hour', 'url')


class TourAdmin(admin.ModelAdmin):
    list_display = ('index', 'place', 'rating', 'review', 'classfications',
                    'address', 'explain', 'mood', 'topic', 'reason', 'cluster')


admin.site.register(Hotel, HotelAdmin)

# admin.site.register(Restaurant, RestaurantAdmin)

admin.site.register(Tour, TourAdmin)
