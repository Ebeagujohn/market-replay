from django.urls import path

from . import views

urlpatterns = [
    path('', views.replay_home, name='replay_home'),
]
