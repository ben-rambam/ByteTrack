# Generated by Django 4.1.1 on 2022-09-08 07:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('timeline', '0004_rename_object_class_trackable_trackable_class'),
    ]

    operations = [
        migrations.AddField(
            model_name='event',
            name='image_path',
            field=models.CharField(default='timeline/placeholder.png', max_length=200),
        ),
    ]