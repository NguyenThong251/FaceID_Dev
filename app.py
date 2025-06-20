from flask import Flask
from routes.routes import erp_face_bp # Import erp_face_bp và report_bp
from flask_cors import CORS
import config.cpu_config as cpu_config  # Import configuration

app = Flask(__name__)

# Danh sách domain frontend được phép truy cập
CORS(app)

# Nếu bạn dùng session/cookie, cần cấu hình thêm:
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = False  # Bắt buộc nếu chạy HTTPS

app.register_blueprint(erp_face_bp, url_prefix='/erp-api-ekyc/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
