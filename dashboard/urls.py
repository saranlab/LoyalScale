from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('upload-csv/', views.upload_csv, name='upload_csv'),
    path('augment-db/', views.augment_db, name='augment_db'),
]

