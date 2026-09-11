// Copy this file to config.js and fill in values from EmailJS Dashboard > Account
// (public key), Email Services (service ID), and Email Templates (template ID).
// EmailJS public keys are designed for browser use; never put private API keys here.
window.VISIONGUIDE_CONFIG = {
  emailjs: {
    publicKey: 'YOUR_PUBLIC_KEY',
    serviceId: 'YOUR_SERVICE_ID',
    templateId: 'YOUR_TEMPLATE_ID',
    volunteerTemplateId: 'YOUR_VOLUNTEER_TEMPLATE_ID',
    emergencyContactEmail: 'trusted-contact@example.com',
  },
};
