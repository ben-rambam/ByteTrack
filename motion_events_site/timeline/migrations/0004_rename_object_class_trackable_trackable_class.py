# Generated by Django 4.1.1 on 2022-09-08 06:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('timeline', '0003_trackableclass_class_id'),
    ]

    operations = [
        migrations.RenameField(
            model_name='trackable',
            old_name='object_class',
            new_name='trackable_class',
        ),
    ]