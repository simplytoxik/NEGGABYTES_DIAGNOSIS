import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from time import sleep
from final_model import load_and_preprocess_data, train_model, predict_diagnosis

# Load and preprocess data, and train the model
filepath = 'riyal_data.csv'  # Ensure this file exists in the working directory
X, y, scaler, label_encoders = load_and_preprocess_data(filepath)
trained_model = train_model(X, y)

# Animation: Spinner class
class Spinner:
    def __init__(self, label):
        self.label = label
        self.running = False
        self.frames = ["|", "/", "-", "\\"]
        self.current_frame = 0

    def start(self):
        self.running = True
        threading.Thread(target=self.animate).start()

    def animate(self):
        while self.running:
            self.label.config(text=self.frames[self.current_frame])
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            sleep(0.1)

    def stop(self):
        self.running = False
        self.label.config(text="")

# Function to validate inputs
def validate_inputs():
    try:
        age = int(age_entry.get())
        if not (1 <= age <= 120):
            raise ValueError("Age must be between 1 and 120.")

        gender = gender_var.get()
        if gender not in ["Male", "Female"]:
            raise ValueError("Select a valid gender.")

        height = float(height_entry.get())
        if height <= 0:
            raise ValueError("Height must be positive.")

        weight = float(weight_entry.get())
        if weight <= 0:
            raise ValueError("Weight must be positive.")

        systolic = int(systolic_entry.get())
        if systolic <= 0:
            raise ValueError("Systolic blood pressure must be positive.")

        diastolic = int(diastolic_entry.get())
        if diastolic <= 0:
            raise ValueError("Diastolic blood pressure must be positive.")

        return age, gender, height, weight, systolic, diastolic
    except ValueError as e:
        messagebox.showerror("Input Error", str(e))
        return None

# Function to handle prediction
def handle_prediction():
    inputs = validate_inputs()
    if not inputs:
        return

    spinner.start()  # Start animation
    predict_button.config(state="disabled")

    def predict():
        try:
            sleep(2)  # Simulate processing delay
            diagnosis = predict_diagnosis(trained_model, scaler, label_encoders, *inputs)
            output_box.insert(tk.END, f"Predicted Diagnosis: {diagnosis}\n")
            output_box.insert(tk.END, "-" * 50 + "\n")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Prediction failed: {e}")
        finally:
            spinner.stop()
            predict_button.config(state="normal")

    threading.Thread(target=predict).start()

# Function to clear inputs with animation
def clear_inputs():
    def fade_out(widget):
        for alpha in range(100, -1, -5):
            widget.config(bg=f"#f{alpha:02x}{alpha:02x}")
            window.update()
            sleep(0.02)
        widget.delete(0, tk.END)

    fade_out(age_entry)
    fade_out(height_entry)
    fade_out(weight_entry)
    fade_out(systolic_entry)
    fade_out(diastolic_entry)
    output_box.delete(1.0, tk.END)
    gender_var.set("Select Gender")

# Main window
window = tk.Tk()
window.title("Medical Diagnosis Predictor")
window.geometry("500x650")

# Styling
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12))
style.configure("TButton", font=("Arial", 12))
style.configure("TEntry", font=("Arial", 12))

# Input fields
ttk.Label(window, text="Age:").grid(row=0, column=0, padx=10, pady=5, sticky="W")
age_entry = ttk.Entry(window, width=30)
age_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(window, text="Gender:").grid(row=1, column=0, padx=10, pady=5, sticky="W")
gender_var = tk.StringVar(value="Select Gender")
gender_menu = ttk.Combobox(window, textvariable=gender_var, values=["Male", "Female"], state="readonly", width=27)
gender_menu.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(window, text="Height (in cm):").grid(row=2, column=0, padx=10, pady=5, sticky="W")
height_entry = ttk.Entry(window, width=30)
height_entry.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(window, text="Weight (in kg):").grid(row=3, column=0, padx=10, pady=5, sticky="W")
weight_entry = ttk.Entry(window, width=30)
weight_entry.grid(row=3, column=1, padx=10, pady=5)

ttk.Label(window, text="Systolic Blood Pressure:").grid(row=4, column=0, padx=10, pady=5, sticky="W")
systolic_entry = ttk.Entry(window, width=30)
systolic_entry.grid(row=4, column=1, padx=10, pady=5)

ttk.Label(window, text="Diastolic Blood Pressure:").grid(row=5, column=0, padx=10, pady=5, sticky="W")
diastolic_entry = ttk.Entry(window, width=30)
diastolic_entry.grid(row=5, column=1, padx=10, pady=5)

# Spinner animation
spinner_label = ttk.Label(window, text="", font=("Arial", 14))
spinner_label.grid(row=6, column=0, columnspan=2, pady=5)
spinner = Spinner(spinner_label)

# Output display
ttk.Label(window, text="Prediction:").grid(row=7, column=0, padx=10, pady=10, sticky="NW")
output_box = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=10, font=("Arial", 10))
output_box.grid(row=7, column=1, padx=10, pady=10, sticky="W")

# Buttons
predict_button = ttk.Button(window, text="Predict", command=handle_prediction)
predict_button.grid(row=8, column=0, padx=10, pady=20)

clear_button = ttk.Button(window, text="Clear", command=clear_inputs)
clear_button.grid(row=9, column=0, padx=1, pady=1, sticky="E")

# Run the Tkinter loop
window.mainloop()
