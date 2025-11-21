from django.urls import path
from . import views

urlpatterns = [
    path("", views.upload_and_visualize, name="upload_and_visualize"),
]
