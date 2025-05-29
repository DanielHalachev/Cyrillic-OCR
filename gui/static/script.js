document.addEventListener('DOMContentLoaded', () => {
  const imageInput = document.getElementById('image-input');
  const fileLabel = document.getElementById('file-label');
  const preview = document.getElementById('preview');
  const previewImg = document.getElementById('preview-img');
  let removeBtn = document.getElementById('remove-btn');
  let submitBtn = document.getElementById('submit-btn');

  // Handle server-side image remove button
  if (removeBtn && previewImg.dataset.source === 'server') {
    removeBtn.addEventListener('click', () => {
      window.location.href = '/';
    });
  }

  // Handle file selection
  if (imageInput) {
    imageInput.addEventListener('change', () => {
      if (imageInput.files.length) {
        const file = imageInput.files[0];
        const previewUrl = URL.createObjectURL(file);
        fileLabel.classList.add('hidden');
        if (preview) {
          preview.classList.remove('hidden');
        }
        previewImg.src = previewUrl;
        previewImg.dataset.source = 'client';

        submitBtn.disabled = false

        // Add or update remove button for client-side preview
        removeBtn = document.getElementById('remove-btn');
        if (!removeBtn) {
          removeBtn = document.createElement('button');
          removeBtn.id = 'remove-btn';
          removeBtn.className = 'absolute top-2 right-2 bg-red-500 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-red-600';
          removeBtn.textContent = '×';
          (preview || previewImg.parentElement).appendChild(removeBtn);
        }
        // Remove any existing listeners by replacing the button
        const newRemoveBtn = removeBtn.cloneNode(true);
        removeBtn.parentNode.replaceChild(newRemoveBtn, removeBtn);
        removeBtn = newRemoveBtn;
        removeBtn.addEventListener('click', () => {
          imageInput.value = '';
          previewImg.src = '#';
          previewImg.dataset.source = 'none';
          if (preview) {
            preview.classList.add('hidden');
          }
          fileLabel.classList.remove('hidden');
          removeBtn.remove();
        });
      }
    });
  }
});