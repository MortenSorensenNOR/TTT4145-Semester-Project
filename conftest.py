from hypothesis import HealthCheck, settings

settings.register_profile("thorough", max_examples=500, deadline=15000, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("default", max_examples=100, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
