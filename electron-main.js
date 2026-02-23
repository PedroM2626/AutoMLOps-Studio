const { app, BrowserWindow, screen } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const os = require('os');
const fs = require('fs');

let mainWindow;
let pythonProcess = null;

// Configuration
const PY_PORT = 8501;
const UI_URL = `http://localhost:${PY_PORT}`;

function getPythonPath() {
  // Simple heuristic to find python. 
  // In a real bundled app, this would point to the bundled executable.
  if (os.platform() === 'win32') {
    return 'python'; // Or check for venv/Scripts/python.exe
  } else {
    return 'python3'; // Or venv/bin/python
  }
}

function createPythonProcess() {
  const pythonExecutable = getPythonPath();
  const scriptPath = path.join(__dirname, 'app.py');
  
  console.log(`Starting Python process: ${pythonExecutable} -m streamlit run ${scriptPath}`);

  // Set environment variable for Hybrid Rendering detection
  const env = { ...process.env, IS_ELECTRON_APP: 'true', PYTHONUNBUFFERED: '1' };

  pythonProcess = spawn(pythonExecutable, [
    '-m', 'streamlit', 'run', scriptPath,
    '--server.port', PY_PORT.toString(),
    '--server.headless', 'true',
    '--server.address', 'localhost' // Security: only bind to localhost
  ], { env });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python]: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error]: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });
}

function createWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  const iconPath = path.join(__dirname, 'assets/icon.png');

  mainWindow = new BrowserWindow({
    width: Math.floor(width * 0.9),
    height: Math.floor(height * 0.9),
    webPreferences: {
      preload: path.join(__dirname, 'electron-preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
    icon: fs.existsSync(iconPath) ? iconPath : undefined,
    title: "AutoMLOps Studio"
  });

  // Load the Streamlit URL
  // We use a retry mechanism to wait for the server
  const loadUrlWithRetry = (retries = 0) => {
    // Simple fetch check or just loadURL and handle fail
    mainWindow.loadURL(UI_URL).catch((err) => {
      console.log(`Server not ready, retrying... (${retries})`);
      if (retries < 20) {
        setTimeout(() => loadUrlWithRetry(retries + 1), 1000);
      } else {
        mainWindow.loadFile(path.join(__dirname, 'error_loading.html')); // Fallback
      }
    });
  };

  loadUrlWithRetry();

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createPythonProcess();
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  if (pythonProcess) {
    if (os.platform() === 'win32') {
        spawn("taskkill", ["/pid", pythonProcess.pid, '/f', '/t']);
    } else {
        pythonProcess.kill('SIGTERM');
    }
  }
});
