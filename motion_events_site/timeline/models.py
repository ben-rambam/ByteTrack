from django.db import models

# Create your models here.

class TrackableClass(models.Model):
    name = models.CharField(max_length=200)
    class_id = models.PositiveSmallIntegerField()

    def __str__(self):
        return self.name


class Trackable(models.Model):
    name = models.CharField(max_length=200)
    trackable_class = models.ForeignKey(TrackableClass, on_delete=models.CASCADE)

    def __str__(self):
        return self.name


class Event(models.Model):
    trackable = models.ForeignKey(Trackable, on_delete=models.CASCADE)
    date = models.DateTimeField()
    class EventType(models.TextChoices):
        APPEARED = 'A', ('Appeared')
        DISAPPEARED = 'D', ('Disappeared')

    event_type = models.CharField(choices=EventType.choices, max_length=2)

    def __str__(self):
        return "Trackable {} {} on {}".format(self.trackable, self.get_event_type_display(), self.date)


class Tracklet(models.Model):
    trackable = models.ForeignKey(Trackable, on_delete=models.CASCADE)
    appeared = models.ForeignKey(Event, related_name='appear_event', on_delete=models.CASCADE)
    disappeared = models.ForeignKey(Event, on_delete=models.CASCADE)
