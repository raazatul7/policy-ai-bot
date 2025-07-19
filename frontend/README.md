# 🌐 Frontend for AI Policy Query System

A beautiful, modern web interface for the AI Policy Query System that allows users to upload insurance policy documents and ask questions in natural language.

## 🚀 Quick Start

### **Option 1: Use the Start Script**
```bash
./start_frontend.sh
```

### **Option 2: Manual Start**
```bash
python3 serve_frontend.py
```

### **Option 3: Open Directly**
Simply open `frontend/index.html` in your web browser.

## 📋 Features

### **🎨 Modern UI/UX**
- Beautiful gradient design
- Responsive layout (works on mobile and desktop)
- Smooth animations and transitions
- Loading indicators and error handling

### **📄 Document Upload**
- Drag and drop file upload
- Support for PDF, DOCX, and TXT files
- File size validation (max 100MB)
- Real-time file type checking

### **❓ Question Interface**
- Large text area for questions
- Example questions for quick start
- Enter key support for quick submission
- Placeholder text with suggestions

### **🤖 AI Integration**
- Real-time connection status to API
- Structured response display
- Error handling and user feedback
- Loading animations during processing

### **📊 Response Display**
- **Decision**: Direct answer to your question
- **Justification**: Detailed explanation from the document
- **Reference**: Specific section/clause reference
- **Document Info**: File details and processing stats

## 🎯 How to Use

1. **Start the API Server** (if not already running):
   ```bash
   cd backend
   python3 main.py
   ```

2. **Start the Frontend**:
   ```bash
   ./start_frontend.sh
   ```

3. **Open your browser** to http://localhost:3000

4. **Upload your policy document** (PDF, DOCX, or TXT)

5. **Ask your question** in natural language

6. **Get instant AI-powered answers!**

## 📱 Example Questions

### **Coverage Questions:**
- "Is maternity coverage included?"
- "What dental services are covered?"
- "Are prescription drugs covered?"

### **Financial Questions:**
- "What is the annual deductible?"
- "What are the copay amounts?"
- "What is the out-of-pocket maximum?"

### **Policy Details:**
- "What is the waiting period for pre-existing conditions?"
- "Are there any exclusions for cosmetic surgery?"
- "How do I file a claim?"

## 🔧 Technical Details

### **API Endpoints Used:**
- `GET /health` - Check server status
- `POST /ask` - Upload document and ask question

### **File Structure:**
```
frontend/
├── index.html          # Main frontend page
└── README.md          # This file
```

### **Dependencies:**
- No external dependencies required
- Pure HTML, CSS, and JavaScript
- Uses Fetch API for HTTP requests

## 🎨 Customization

### **Colors:**
The frontend uses a purple gradient theme. You can customize colors by modifying the CSS variables in `index.html`:

```css
/* Main gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Accent color */
color: #667eea;
```

### **API URL:**
To change the API server URL, modify the `API_BASE_URL` constant in the JavaScript:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## 🐛 Troubleshooting

### **Frontend Won't Load:**
- Make sure you're running the script from the project root
- Check that port 3000 is available
- Try opening `frontend/index.html` directly in your browser

### **Can't Connect to API:**
- Make sure the API server is running on localhost:8000
- Check the connection status indicator on the page
- Verify the API server is healthy at http://localhost:8000/health

### **File Upload Issues:**
- Ensure your file is PDF, DOCX, or TXT format
- Check file size (max 100MB)
- Try a different browser if issues persist

## 🚀 Production Deployment

For production deployment, you can:

1. **Use a proper web server** (nginx, Apache)
2. **Serve static files** from the `frontend/` directory
3. **Configure CORS** if needed
4. **Add HTTPS** for security

## 📄 License

This frontend is part of the AI Policy Query System and follows the same license terms.

---

**🎯 Ready to use? Start the frontend and ask questions about your insurance policies!** 