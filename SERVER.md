Below are the SSH credentials for the development server:
- Host: ssh -p 36722 ravi@public.cht77.com  
- Username: 
- Password: 
Please log in and change your password after your first login if prompted. I also added you to the sudo list, so please use this carefully if any change needs to be made. 
This server will be used to develop and deploy the new FarmAI web service. Our production domain will eventually be:
https://farmai.cht77.com (not yet set up)
During development, however, you are welcome to access and test your application using the server's IP address or another temporary URL while you are getting everything configured.
The goal is for you to become familiar with setting up and managing a web application on an Ubuntu server. This includes tasks such as:
- Setting up your project directory
- Creating a Python virtual environment (if needed)
- Installing the required packages and dependencies
- Configuring and running your application
- Connecting Apache to your application
- Testing that the web service is accessible through a browser
There are two apps hosted by Apache server for now, I think you can use the similar architecture for this Farm app, so we can set up a virtual host and redirect to this app. 
Feel free to explore different deployment approaches and learn how the various components work together. The server is intended to be a hands-on learning environment.
If you need any software installed, additional permissions, a database configured, Apache updated, or help troubleshooting any part of the setup, just let me know. I'm happy to help whenever you get stuck.