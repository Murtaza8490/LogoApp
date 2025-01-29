# logo_app.py
import torch
import numpy as np
from diffusers import DiffusionPipeline
from PIL import Image, ImageTk
import potrace  # Corrected import from pypotrace to potrace
import tkinter as tk
from tkinter import ttk, filedialog, Canvas
import zipfile  # Added missing import

class LogoGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoLogo Designer Pro")
        self.root.geometry("800x600")
        
        # Initialize models and presets
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            "SDXL 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
            "Playground v2.5": "playgroundai/playground-v2.5-1024px-aesthetic",
            "SSD-1B": "Segmind/SSD-1B"
        }
        
        self.style_presets = {
            "Modern": "clean lines, minimalistic, geometric shapes, gradient colors",
            "Vintage": "hand-drawn, textured, distressed look, muted colors",
            "Tech": "neon glow, circuit patterns, dark background, futuristic",
            "Organic": "flowing shapes, natural colors, hand-drawn elements",
            "Corporate": "bold typography, solid colors, professional layout"
        }
        
        self.current_pipe = None
        self.current_image = None
        self.create_widgets()
        
    def create_widgets(self):
        """Create GUI components with preview and presets"""
        # Control Panel
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Model Selection
        ttk.Label(control_frame, text="Model:").pack(anchor=tk.W)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            control_frame, 
            textvariable=self.model_var,
            values=list(self.models.keys()),
            state="readonly"
        )
        self.model_dropdown.current(0)
        self.model_dropdown.pack(pady=5, fill=tk.X)
        
        # Style Presets
        ttk.Label(control_frame, text="Style Preset:").pack(anchor=tk.W)
        self.style_var = tk.StringVar()
        self.style_dropdown = ttk.Combobox(
            control_frame, 
            textvariable=self.style_var,
            values=list(self.style_presets.keys()),
            state="readonly"
        )
        self.style_dropdown.current(0)
        self.style_dropdown.pack(pady=5, fill=tk.X)
        
        # Prompt Input
        ttk.Label(control_frame, text="Design Prompt:").pack(anchor=tk.W)
        self.prompt_entry = ttk.Entry(control_frame, width=30)
        self.prompt_entry.pack(pady=5, fill=tk.X)
        self.prompt_entry.insert(0, "tech company logo")
        
        # Generate Button
        self.generate_btn = ttk.Button(
            control_frame, 
            text="Generate Logo",
            command=self.generate_logo
        )
        self.generate_btn.pack(pady=10, fill=tk.X)
        
        # Preview Panel
        preview_frame = ttk.Frame(self.root, padding=10)
        preview_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.canvas = Canvas(preview_frame, bg='white')
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def initialize_model(self):
        """Load selected diffusion model"""
        model_name = self.models[self.model_var.get()]
        
        if self.current_pipe:
            del self.current_pipe
            torch.cuda.empty_cache()

        self.status_var.set(f"Loading {model_name}...")
        self.root.update_idletasks()
        
        self.current_pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(self.device)
        
        # Optimize for lower VRAM
        self.current_pipe.enable_attention_slicing()
        
    def generate_logo(self):
        """Handle logo generation with style presets"""
        try:
            if not self.current_pipe:
                self.initialize_model()
                
            # Build enhanced prompt
            base_prompt = self.prompt_entry.get()
            style_addition = self.style_presets[self.style_var.get()]
            full_prompt = (
                f"Professional vector logo design: {base_prompt} "
                f"in {self.style_var.get().lower()} style:: "
                f"{style_addition}::2"
            )
            
            # Generate image
            self.status_var.set("Generating logo...")
            self.root.update_idletasks()
            
            image = self.current_pipe(
                prompt=full_prompt,
                negative_prompt="low quality, blurry, text, watermark",
                num_inference_steps=30,
                guidance_scale=8.0
            ).images[0]
            
            # Update preview
            self.update_preview(image)
            self.status_var.set("Generation complete!")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            torch.cuda.empty_cache()
            
    def update_preview(self, image):
        """Update preview window with new image"""
        # Convert PIL Image to Tkinter PhotoImage
        preview_image = image.resize((400, 400))
        self.tk_image = ImageTk.PhotoImage(preview_image)
        
        # Clear canvas and draw new image
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width()/2,
            self.canvas.winfo_height()/2,
            image=self.tk_image,
            anchor=tk.CENTER
        )
        
        # Save reference to prevent garbage collection
        self.current_image = image
        
    def vectorize_image(self, image):
        """Convert PIL Image to SVG with style-based parameters"""
        try:
            # Style-based threshold adjustments
            style = self.style_var.get()
            threshold = 128  # Default
            if style == "Vintage":
                threshold = 100
            elif style == "Tech":
                threshold = 160
            
            img_gray = image.convert('L')
            bitmap = potrace.Bitmap(np.array(img_gray) < threshold)  # Corrected potrace usage
            path = bitmap.trace()
            
            svg_content = f'<svg width="{image.width}" height="{image.height}">'
            for curve in path:
                svg_content += f'<path d="{curve}" fill="black"/>'  # Corrected path data extraction
            svg_content += '</svg>'
            
            return svg_content
        except Exception as e:
            self.status_var.set(f"Vectorization error: {str(e)}")
            return None
            
    def save_files(self):
        """Save generated assets with style-based naming"""
        if not self.current_image:
            return
            
        # Get save path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP Archive", "*.zip")],
            initialfile=f"logo_{self.style_var.get().lower()}"
        )
        
        if file_path:
            # Generate SVG
            svg_content = self.vectorize_image(self.current_image)
            
            # Save files
            self.current_image.save("logo.png")
            with open("logo.svg", "w") as f:
                if svg_content:
                    f.write(svg_content)
            
            # Create package
            with zipfile.ZipFile(file_path, 'w') as zipf:
                zipf.write("logo.png")
                if svg_content:
                    zipf.write("logo.svg")
            
            self.status_var.set(f"Saved package: {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LogoGeneratorApp(root)
    root.mainloop()