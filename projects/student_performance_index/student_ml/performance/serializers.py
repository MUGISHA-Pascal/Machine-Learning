from rest_framework import serializers
from .models import StudentPerformance

class StudentPerformanceSerializers(serializers.ModelSerializer):
    class Meta:
        models = StudentPerformance
        fields = '__all__'