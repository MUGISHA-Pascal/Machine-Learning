from django.db import models

# Create your models here.
class StudentPerformance(models.Model):
    hours_studied = models.IntegerField()
    previous_scores = models.IntegerField()
    extracurricular = models.BooleanField()
    sleep_hours  = models.IntegerField()
    sample_papers = models.IntegerField()
    performance_index = models.FloatField()

    def __str__(self):
        return f"Performance: {self.performance_index}"