# Generated by Django 3.0.5 on 2023-05-20 18:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hospital', '0002_auto_20230520_0458'),
    ]

    operations = [
        migrations.AddField(
            model_name='patientprescriptiondata',
            name='cropped_image',
            field=models.ImageField(blank=True, upload_to='prescriptions/cropped_images'),
        ),
    ]
