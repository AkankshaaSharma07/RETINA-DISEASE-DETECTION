import os
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import cv2
from UNET.model import build_unet
from UNET.utils import seeding
from torchvision import transforms  # For image transformations
from train_classification import CombinedImageClassifier  # Import your classification model
from train_classification import classes #Import classes names
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'visionai2999@gmail.com'  # Replace with your Gmail
app.config['MAIL_PASSWORD'] = 'fzcj seia vdwb noii'  # Replace with your Gmail app password

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
app.secret_key = 'secretkey'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    mobile_number = db.Column(db.String(15), nullable=False)
    _password = db.Column('password', db.String(100), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    reset_otp = db.Column(db.String(6), nullable=True)
    reset_otp_expiry = db.Column(db.DateTime, nullable=True)

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, value):
        if not value:
            raise ValueError('Password cannot be empty')
        self._password = bcrypt.hashpw(value.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        try:
            return bcrypt.checkpw(password.encode('utf-8'), self._password.encode('utf-8'))
        except ValueError:
            # If the stored password is not properly hashed, try to rehash it
            try:
                # Try to verify with the old password
                if self._password == password:
                    # If it matches, rehash it properly
                    self.password = password
                    db.session.commit()
                    return True
                return False
            except:
                return False

def migrate_passwords():
    """Migrate existing passwords to proper bcrypt hashing"""
    with app.app_context():
        users = User.query.all()
        for user in users:
            try:
                # If the password is not properly hashed, it will raise ValueError
                bcrypt.checkpw(b'test', user._password.encode('utf-8'))
            except ValueError:
                # The password is not properly hashed, we need to fix it
                try:
                    # Try to verify with the current password
                    if user.check_password(user._password):
                        print(f"Successfully migrated password for user {user.email}")
                    else:
                        print(f"Could not migrate password for user {user.email}")
                except:
                    print(f"Error migrating password for user {user.email}")

# Load your trained model
try:
    model = load_model('model/best_model.keras')
    print("Model loaded successfully!")
    # Print model summary to see expected input shape
    model.summary()
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Model loading error: {str(e)}")
    MODEL_LOADED = False
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image1(image_path):
    try:
        # Read image using cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
            
        # Print original image shape for debugging
        print(f"Original image shape: {img.shape}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 256x256 as expected by the model
        img_resized = cv2.resize(img, (256, 256))
        print(f"Resized image shape: {img_resized.shape}")
        
        # Convert to float32 (no normalization)
        img_normalized = img_resized.astype(np.float32)
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        print(f"Final input shape: {img_batch.shape}")
        
        return img_batch
            
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

def analyze_image(image_path):
    if not MODEL_LOADED:
        return {
            'condition': 'Model not loaded',
            'confidence': 0,
            'recommendations': 'System is not ready. Please contact support.'
        }
    
    try:
        # Preprocess the image
        processed_img = preprocess_image1(image_path)
        
        # Print shapes for debugging
        print(f"Input shape to model: {processed_img.shape}")
        
        # Get prediction from model
        prediction = model.predict(processed_img, verbose=1)
        print(f"Model prediction shape: {prediction.shape}")
        print(f"Raw prediction values: {prediction}")
        
        # Convert prediction to human-readable format
        conditions = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
        
        # Print probabilities for each class
        print("\nProbabilities for each class:")
        for i, condition in enumerate(conditions):
            prob = prediction[0][i] * 100
            print(f"{condition}: {prob:.2f}%")
        
        predicted_class = conditions[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100
        
        print(f"\nFinal Prediction:")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Generate recommendations based on prediction
        if predicted_class == 'Normal':
            recommendations = "Your retinal scan appears normal. Regular check-up recommended in 6 months."
        elif predicted_class == 'Diabetic Retinopathy':
            recommendations = "Signs of Diabetic Retinopathy detected. Please schedule an appointment with an ophthalmologist within 1 week. Early treatment can prevent vision loss."
        elif predicted_class == 'Glaucoma':
            recommendations = "Indicators of Glaucoma observed. Urgent consultation with an ophthalmologist required. Early treatment is crucial for preventing vision loss."
        else:  # Cataract
            recommendations = "Signs of Cataract detected. Schedule an appointment with an eye surgeon to discuss treatment options. Modern cataract surgery is safe and effective."
        
        return {
            'condition': predicted_class,
            'confidence': round(confidence, 2),
            'recommendations': recommendations
        }
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        error_msg = str(e)
        if "Could not read image file" in error_msg:
            return {
                'condition': 'Error reading image',
                'confidence': 0,
                'recommendations': 'Please ensure the image file is valid and try again'
            }
        elif "Input shape" in error_msg or "Dimension" in error_msg:
            return {
                'condition': 'Model input error',
                'confidence': 0,
                'recommendations': 'Image preprocessing failed. Please contact support with error: ' + str(e)
            }
        return {
            'condition': 'Error in analysis',
            'confidence': 0,
            'recommendations': f'An error occurred: {str(e)}. Please try again or contact support.'
        }

@app.route('/')
def home():
    return render_template('Landingpage.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        # If user is already logged in, redirect to landing page
        if 'email' in session:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'redirect': url_for('Landingpage')})
            return redirect(url_for('Landingpage'))
            
        if request.method == 'POST':
            try:
                email = request.form.get('email', '').strip()
                password = request.form.get('password', '')
                
                # Basic validation
                if not email or not password:
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'error': 'Please enter both email and password'}), 400
                    flash('Please enter both email and password', 'error')
                    return render_template('login.html')
                    
                # Find user by email
                user = User.query.filter_by(email=email).first()
                
                # Strict check for user existence - must register first
                if not user:
                    error_message = 'Account not found. Please register first.'
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({
                            'error': error_message,
                            'redirect': url_for('register')
                        }), 401
                    flash(error_message, 'error')
                    return redirect(url_for('register'))
                    
                # Check password
                if not user.check_password(password):
                    error_message = 'Invalid password'
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'error': error_message}), 401
                    flash(error_message, 'error')
                    return render_template('login.html')
                    
                # Login successful
                session.clear()  # Clear any existing session data
                session['email'] = user.email
                session['user_id'] = user.id
                session['name'] = user.name
                
                # Log successful login
                print(f"User {email} logged in successfully")
                
                # Handle AJAX request
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': True,
                        'redirect': url_for('Landingpage'),
                        'message': 'Login successful!'
                    })
                
                # Handle regular form submission
                flash('Welcome back!', 'success')
                return redirect(url_for('Landingpage'))
                    
            except Exception as e:
                db.session.rollback()
                error_message = f'An unexpected error occurred during login (POST): {str(e)}'
                print(f"Login exception (POST): {error_message}") # Log the specific error

                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': 'An error occurred during login. Please try again.'}), 500
                flash('An error occurred during login', 'error')
                return render_template('login.html')
                
        # Handle GET request for the login page
        return render_template('login.html')

    except Exception as e:
        db.session.rollback()
        error_message = f'An unexpected error occurred while rendering login page: {str(e)}'
        print(f"Login exception (GET/Rendering): {error_message}") # Log the specific error
        # Attempt to render a simple error message or template
        try:
            return "An internal server error occurred.", 500
        except:
            return "An internal server error occurred and couldn't render error page.", 500

@app.route('/Landingpage')
def Landingpage():
    return render_template('Landingpage.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            mobile_number = request.form.get('mobile_number', '').strip()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            gender = request.form.get('gender', '')
            age = request.form.get('age', '')
            
            # Validate required fields
            if not all([name, email, mobile_number, password, gender, age]):
                error_message = 'All fields are required'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': error_message}), 400
                flash(error_message, 'error')
                return render_template('register.html')
            
            # Validate email format
            if '@' not in email or '.' not in email:
                error_message = 'Invalid email format'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': error_message}), 400
                flash(error_message, 'error')
                return render_template('register.html')
            
            # Validate mobile number format
            # Allowing 10-15 digits for mobile number based on common formats
            if not mobile_number.isdigit() or not (10 <= len(mobile_number) <= 15):
                error_message = 'Invalid mobile number format (10-15 digits required)'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': error_message}), 400
                flash(error_message, 'error')
                return render_template('register.html')
            
            # Validate age
            try:
                age = int(age)
                if age < 1 or age > 120:
                    raise ValueError("Age must be between 1 and 120")
            except ValueError:
                error_message = 'Invalid age'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': error_message}), 400
                flash(error_message, 'error')
                return render_template('register.html')
            
            # Validate password
            if len(password) < 8:
                error_message = 'Password must be at least 8 characters long'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': error_message}), 400
                flash(error_message, 'error')
                return render_template('register.html')
            
            # Check if passwords match
            if password != confirm_password:
                error_message = 'Passwords do not match'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'error': error_message}), 400
                flash(error_message, 'error')
                return render_template('register.html')
            
            # Check if user already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                error_message = 'Email already registered. Please login instead.'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'error': error_message,
                        'redirect': url_for('login')
                    }), 400
                flash(error_message, 'error')
                return redirect(url_for('login'))
            
            # Create new user
            user = User(
                name=name,
                email=email,
                mobile_number=mobile_number,
                gender=gender,
                age=age
            )
            user.password = password  # This will automatically hash the password
            
            db.session.add(user)
            db.session.commit()
            
            success_message = 'Registration successful! Please login.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': True, 
                    'message': success_message,
                    'redirect': url_for('login')
                })
            
            flash(success_message, 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            error_message = f'An error occurred during registration: {str(e)}'
            print(f"Registration error: {error_message}")
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': 'An internal error occurred during registration. Please try again.'}), 500 # Generic error for client
            
            flash('An error occurred during registration', 'error')
            return render_template('register.html')
            
    return render_template('register.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/test')
def test():
    if 'email' not in session:
        flash('Please login to access the analysis page', 'error')
        return redirect(url_for('login'))
    return render_template('test.html')

@app.route('/test1')
def test1():
    return render_template('test1.html')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        flash('Please login to access the dashboard', 'error')
        return redirect(url_for('login'))
    
    user = User.query.filter_by(email=session['email']).first()
    if not user:
        session.clear()
        flash('User not found. Please login again.', 'error')
        return redirect(url_for('login'))
        
    return render_template('dashboard.html', user=user)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('home'))

@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    if 'retina_image' not in request.files:
        return render_template('test.html', error='No file uploaded')
    
    file = request.files['retina_image']
    if file.filename == '':
        return render_template('test.html', error='No file selected')
    
    if file and allowed_file(file.filename):
        # Here you would add your AI model processing logic
        # For now, we'll just return a success message
        return render_template('test.html', success='Image uploaded successfully! Analysis in progress...')
    
    return render_template('test.html', error='Invalid file type')

@app.route('/analyze_image', methods=['POST'])
def analyze_image_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the image
            result = analyze_image(filepath)
            
            # Add the image path to the result for display
            result['image_path'] = url_for('static', filename=f'uploads/{filename}')
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400



# Seeding
seeding(42)

# Load the UNet model for segmentation
unet_checkpoint_path = 'files/checkpoint.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unet_model = build_unet()
unet_model = unet_model.to(device)
unet_model.load_state_dict(torch.load(unet_checkpoint_path, map_location=device))
unet_model.eval()

H, W = 512, 512

# Load the classification model
classification_model_path = 'pytorch_model.pth'  # Replace with your model path
classification_model = CombinedImageClassifier(len(classes)).to(device)  # Initialize the model
classification_model.load_state_dict(torch.load(classification_model_path, map_location=device))  # Load state dict
classification_model.eval()  # Set to evaluation mode
classification_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])  # Preprocessing steps for classification


def preprocess_image(image_path):
    """ Read and preprocess the image for the UNet model """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # (H, W, 3)
    image = cv2.resize(image, (W, H))
    x = np.transpose(image, (2, 0, 1))  # (3, H, W)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)  # (1, 3, H, W)
    x = x.astype(np.float32)
    x = torch.from_numpy(x).to(device)
    return x


def predict_disease(retinal_image_path, mask_image_path):
    """Predicts the disease based on the retinal and mask images using classification_model"""
    retinal_image = Image.open(retinal_image_path).convert("RGB")
    mask_image = Image.open(mask_image_path).convert("L")
    mask_image = Image.fromarray(np.uint8(np.array(mask_image) * 255))

    retinal_tensor = classification_transform(retinal_image)
    mask_tensor = classification_transform(mask_image)
    mask_tensor = mask_tensor.expand(3, -1, -1)

    combined_image = torch.cat((retinal_tensor, mask_tensor), dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classification_model(combined_image)
        _, predicted = torch.max(output, 1)
        disease = classes[predicted.item()]

    return disease

@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    result_image_path = None
    disease_prediction = None  # To store the disease prediction
    file_path = None  # To store the original image path
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Perform UNet inference for segmentation
            x = preprocess_image(file_path)

            with torch.no_grad():
                pred_y = unet_model(x)
                pred_y = torch.sigmoid(pred_y)
                pred_y = pred_y[0].cpu().numpy()
                pred_y = np.squeeze(pred_y, axis=0)  # (H, W)
                pred_y = (pred_y > 0.5).astype(np.uint8) * 255

            # Save the generated mask
            result_path = os.path.join(app.config['RESULT_FOLDER'], 'masked_' + file.filename)
            cv2.imwrite(result_path, pred_y)

            result_image_path = result_path

            # Perform classification using both original and masked image
            disease_prediction = predict_disease(file_path, result_path)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
            print(f"Original image path: {file_path} {result_image_path}")
        

    return render_template('segmentation.html', result_image_path=result_image_path, disease_prediction=disease_prediction, original_image_path=file_path)

def send_reset_email(email, otp):
    try:
        msg = MIMEMultipart()
        msg['From'] = app.config['MAIL_USERNAME']
        msg['To'] = email
        msg['Subject'] = "Password Reset OTP"
        
        body = f"""
        Your OTP for password reset is: {otp}
        Please use this OTP to reset your password. The OTP is valid for 10 minutes.
        If you did not request this, please ignore this email.
        
        Thank you,
        VisionAI Team
        """
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'])
        server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        text = msg.as_string()
        server.sendmail(app.config['MAIL_USERNAME'], email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate OTP
            otp = ''.join(random.choices(string.digits, k=6))
            user.reset_otp = otp
            user.reset_otp_expiry = datetime.now() + timedelta(minutes=10)
            
            try:
                db.session.commit()
                if send_reset_email(email, otp):
                    flash('OTP has been sent to your email address.', 'success')
                    return redirect(url_for('verify_otp', email=email))
                else:
                    flash('Error sending OTP. Please try again.', 'error')
            except Exception as e:
                db.session.rollback()
                flash('An error occurred. Please try again.', 'error')
        else:
            flash('Email address not found.', 'error')
    
    return render_template('forgot_password.html')

@app.route('/verify-otp/<email>', methods=['GET', 'POST'])
def verify_otp(email):
    if request.method == 'POST':
        otp = request.form.get('otp')
        user = User.query.filter_by(email=email).first()
        
        if user and user.reset_otp == otp and user.reset_otp_expiry > datetime.now():
            # Clear OTP once verified
            user.reset_otp = None
            user.reset_otp_expiry = None
            db.session.commit()
            
            # Store email in session for password reset
            session['reset_email'] = email
            return redirect(url_for('reset_password'))
        else:
            flash('Invalid or expired OTP.', 'error')
    
    return render_template('verify_otp.html', email=email)

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_email' not in session:
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html')
        
        user = User.query.filter_by(email=session['reset_email']).first()
        if user:
            user.password = password
            db.session.commit()
            session.pop('reset_email', None)
            flash('Password has been reset successfully.', 'success')
            return redirect(url_for('login'))
    
    return render_template('reset_password.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        migrate_passwords()  # Run the migration
    app.run(debug=False)