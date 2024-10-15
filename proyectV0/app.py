import tkinter as tk
from registration import RegistrationScreen
from login import LoginScreen

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("400x300")
        self.root.title("Sistema de Reconocimiento Facial")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack()

        self.btn_register = tk.Button(self.main_frame, text="Registrarse", width=40, command=self.open_registration)
        self.btn_register.pack(pady=10)

        self.btn_login = tk.Button(self.main_frame, text="Iniciar Sesión", width=40, command=self.open_login)
        self.btn_login.pack(pady=10)

    def open_registration(self):
        self.main_frame.pack_forget()
        self.root.geometry("800x600")
        RegistrationScreen(self.root, self)

    def open_login(self):
        self.main_frame.pack_forget()
        self.root.geometry("800x600")
        LoginScreen(self.root, self)

    def show_main_menu(self):
        self.main_frame.pack()
        self.root.geometry("400x300")

if __name__ == '__main__':
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.mainloop()
