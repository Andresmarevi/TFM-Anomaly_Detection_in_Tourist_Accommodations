from dataclasses import dataclass


@dataclass
class Alojamiento:
    hotel_id: int = 0
    room_id: int = 0
    room_name: str = ""
    hotel_name: str = ""
    location_id: str = ""
    accommodation_type: str = ""
    num_bedrooms: int = 0
    num_bathrooms: int = 0
    area: str = ""
    price: float = 0.0
    offer: float = 0.0
    price_without_offer: float = 0.0
    capacity: int = 0
    meal_plan: str = ""
    booking_conditions: str = ""
    stars: str = ""
    longitude: str = ""
    latitude: str = ""
    piscina = 0
    primera_linea = 0
