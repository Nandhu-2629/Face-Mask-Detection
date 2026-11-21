from django.db import models

class Detection(models.Model):

    result = models.CharField(max_length=20)

    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.result