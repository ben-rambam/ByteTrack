from django.contrib import admin

# Register your models here.

from .models import TrackableClass, Trackable, Event, Tracklet

admin.site.register(TrackableClass)
admin.site.register(Trackable)
admin.site.register(Event)
admin.site.register(Tracklet)
