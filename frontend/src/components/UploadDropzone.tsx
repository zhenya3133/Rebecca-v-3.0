import React, { useCallback, useState } from "react";

type UploadDropzoneProps = {
  onFilesSelected?: (files: FileList) => Promise<void> | void;
};

export const UploadDropzone: React.FC<UploadDropzoneProps> = ({ onFilesSelected }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      const files = event.dataTransfer.files;
      if (files && files.length && onFilesSelected) {
        await onFilesSelected(files);
      }
    },
    [onFilesSelected],
  );

  const handleFileChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files;
      if (files && files.length && onFilesSelected) {
        await onFilesSelected(files);
      }
    },
    [onFilesSelected],
  );

  return (
    <div
      className={`dropzone ${isDragging ? "dropzone--active" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <p>Drag & drop documents here or click to browse</p>
      <input type="file" multiple className="dropzone-input" onChange={handleFileChange} />
    </div>
  );
};
