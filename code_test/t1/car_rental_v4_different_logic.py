# Car Rental System - Advanced Version (Different Logic)

import json
import os

class Car:
    def __init__(self, name, rate, count):
        self.name = name
        self.rate = rate
        self.count = count

class RentalSystem:
    def __init__(self):
        self.data_file = "rental_data.json"
        self.cars = self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as f:
                data = json.load(f)
                return [Car(**item) for item in data]
        return [
            Car("Toyota", 80, 3),
            Car("Honda", 100, 2),
            Car("Suzuki", 60, 4)
        ]

    def save_data(self):
        with open(self.data_file, "w") as f:
            json.dump([c.__dict__ for c in self.cars], f, indent=2)

    def show_available(self):
        print("\nAvailable Cars:")
        for c in self.cars:
            print(f"{c.name}: {c.count} cars, ${c.rate}/day")

    def rent_car(self, name, days):
        for c in self.cars:
            if c.name.lower() == name.lower() and c.count > 0:
                c.count -= 1
                self.save_data()
                print(f"You rented {c.name} for {days} days. Total: ${c.rate * days}")
                return
        print("Car unavailable or invalid name.")

    def return_car(self, name):
        for c in self.cars:
            if c.name.lower() == name.lower():
                c.count += 1
                self.save_data()
                print(f"Returned {c.name}. Thank you!")
                return
        print("Car not found in system.")

if __name__ == "__main__":
    sys = RentalSystem()
    while True:
        print("\nMenu: [1] View [2] Rent [3] Return [4] Exit")
        choice = input("Select: ")
        if choice == "1":
            sys.show_available()
        elif choice == "2":
            car_name = input("Car name: ")
            days = int(input("Days: "))
            sys.rent_car(car_name, days)
        elif choice == "3":
            car_name = input("Return car name: ")
            sys.return_car(car_name)
        elif choice == "4":
            break
        else:
            print("Invalid choice.")
