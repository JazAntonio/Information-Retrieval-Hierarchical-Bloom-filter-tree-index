from Application_Layer import Service


class Interface:
    @staticmethod
    def menu():
        print("1. Add a new document file")
        print("2. Remove a document")
        print("3. Perform a query search")
        print("4. Exit")

    @staticmethod
    def get_option():
        try:
            return int(input("Choose an option: "))
        except ValueError:
            return -1


def main():
    interface = Interface()
    service = Service(28)
    service.index_structure.initialization()

    while True:
        interface.menu()
        option = interface.get_option()

        if option == 1:

            try:
                service.insert_document()
            except ValueError:
                continue

        elif option == 2:

            # service.delete_document()
            print("Service not available.")

        elif option == 3:

            service.query_search()

        elif option == 4:
            break
        else:
            print("Invalid option, choose again.")


if __name__ == "__main__":
    main()
