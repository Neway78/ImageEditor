#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QDialogButtonBox>

#include <iostream>

#include "popup.h"

PopUp::PopUp(QStringList *popupFields, QStringList *defaultValues, QWidget *parent)
    : QDialog(parent)
{
    QFormLayout *form = new QFormLayout(this);
    for (int i = 0; i < popupFields->size(); i++) {
        QLabel *iLabel = new QLabel(popupFields->at(i), this);
        QLineEdit *iLine = new QLineEdit(this);
        iLine->setAlignment(Qt::AlignRight);
        iLine->setText(defaultValues->at(i));
        form->addRow(iLabel, iLine);
        
        fields << iLine;
    }

    QDialogButtonBox *buttonBox = new QDialogButtonBox
            ( QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
              Qt::Horizontal, this );
    form->addWidget(buttonBox);
    setLayout(form);

    this->setAttribute(Qt::WA_DeleteOnClose);
    
    bool conn = connect(buttonBox, &QDialogButtonBox::accepted,
                        this, &PopUp::accepted);
    Q_ASSERT(conn);
    conn = connect(buttonBox, &QDialogButtonBox::rejected, 
                   this, &PopUp::rejected);
    Q_ASSERT(conn);
    conn = connect(this, SIGNAL(transferFields(QStringList)),
                   parent, SLOT(retrieveFields(QStringList)));
    Q_ASSERT(conn);
}

void PopUp::accepted()
{
    QStringList outputFields;
    if (!outputFields.isEmpty()) {
        outputFields.clear();
    }
    foreach (QLineEdit* lineEdit, fields) {
        outputFields.append(lineEdit->text());
    }
    emit(transferFields(outputFields));
    this->close();
}

void PopUp::rejected()
{
    this->close();
}