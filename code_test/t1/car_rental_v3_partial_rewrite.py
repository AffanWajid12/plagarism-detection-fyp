# Car Rental System - Refactored Version (Partially Rewritten)

class CarRental:
    def __init__(self):
        self.cars = {
            "Toyota": {"qty": 3, "price": 80},
            "Honda": {"qty": 2, "price": 100},
            "Suzuki": {"qty": 4, "price": 60}
        }
        self.rented = {}

    def list_cars(self):
        print("\n--- Car List ---")
        for model, info in self.cars.items():
            print(f"{model}: {info['qty']} left, ${info['price']} per day")

    def rent(self, model, days):
        if model in self.cars and self.cars[model]["qty"] > 0:
            total = self.cars[model]["price"] * days
            self.cars[model]["qty"] -= 1
            self.rented[model] = days
            print(f"{model} rented for {days} days. Total: ${total}")
        else:
            print("Unavailable or invalid car.")

    def return_car(self, model):
        if model in self.rented:
            self.cars[model]["qty"] += 1
            del self.rented[model]
            print(f"{model} returned. Thank you!")
        else:
            print("This car wasn’t rented.")

def main():
    system = CarRental()
    while True:
        print("\n1. View\n2. Rent\n3. Return\n4. Exit")
        ch = input("Choice: ")
        if ch == "1":
            system.list_cars()
        elif ch == "2":
            system.list_cars()
            model = input("Car name: ")
            days = int(input("Days: "))
            system.rent(model, days)
        elif ch == "3":
            model = input("Car to return: ")
            system.return_car(model)
        elif ch == "4":
            break
        else:
            print("Invalid.")

if __name__ == "__main__":
    main()
