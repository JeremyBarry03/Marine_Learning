# B351-final-project
## How to run this code
To run this code, a user is required to start up the front and backend code

Start by navigating to the directory where app.py is in Terminal on Mac
Run the following command

python3 app.py

Start the frontend by navigating to the directory where index.html is in Terminal on Mac
Run the following command

python3 -m http.server

Navigate to the link in your browser
http://127.0.0.1:8000/ 

## Web Application Development Tools

### CSS and SCSS
- **Using CSS:** You can opt for plain CSS. If so do not add to main.css it will be overridden when running the Makefile.
- **Using SCSS:** Ensure you have an SCSS compiler installed. SCSS offers more features like variables, nesting, and mixins which can make your CSS more maintainable. (and clean)

### SCSS Compiler Setup
1. **Install SCSS Compiler:** To compile SCSS, you first need to install an SCSS compiler. You can find installation instructions on the [official SASS website](https://sass-lang.com/install/).
2. **Compile SCSS:**
   - On VSCode you can right click and open the in integrated Terminal.
   - Make sure you are in root directory of the project.
   - Run the command `make` to compile your SCSS files into CSS.

## Build Tools
When working on this code, there a couple libraries that need to be installed first.

Use the following commands in terminal...
pip install tensorflow
pip install opencv-pyton
pip install Flask

- **Make:** A build automation tool that automatically builds executable programs and libraries from source code by reading files called Makefiles which specify how to derive the target program.
- **Installing Make:**
   - For macOS, you can install Make using Homebrew with the command: `brew install make`.
