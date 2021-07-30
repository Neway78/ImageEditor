#ifndef POPUP_H
#define POPUP_H

#include <QWidget>
#include <QDialog>
#include <QStringList>
#include <QLineEdit>

class PopUp : public QDialog
{
    Q_OBJECT;

public:
    PopUp(QStringList *popupFields, QStringList *defaultValues, QWidget *parent = nullptr);

signals:
    void transferFields(QStringList);

private slots:
    void accepted();
    void rejected();

private:
    QList<QLineEdit*> fields;
};

#endif
